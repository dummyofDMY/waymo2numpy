import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils
from collections import defaultdict
import os, cv2

segmentation_3d_mapping_dic = {
    8: 2,
    9: 3,
    10: 1,
    14: 4,
    15: 8,
    16: 8,
    17: 5,
    18: 5,
    19: 6,
    20: 6,
    22: 7
}
segmentation_3d_mapping_dic = defaultdict(lambda: 0, segmentation_3d_mapping_dic)
lookup_table_3d = np.zeros(23, dtype=np.int32)
for i in range(23):
    lookup_table_3d[i] = segmentation_3d_mapping_dic[i]


segmentation_2d_mapping_dic = {
    15: 1,
    17: 2,
    18: 3,
    19: 4,
    20: 5,
    21: 6,
    22: 6,
    23: 7,
    24: 8,
}
segmentation_2d_mapping_dic = defaultdict(lambda: 0, segmentation_2d_mapping_dic)
lookup_table_2d = np.zeros(29, dtype=np.int32)
for i in range(29):
    lookup_table_2d[i] = segmentation_2d_mapping_dic[i]


if __name__ == "__main__":
    count = 0

    target_dir = '/home/yiwei-guest/code/waymo/download/training_0002'  # 4
    if not os.path.exists(target_dir):
        print(f"Target directory {target_dir} does not exist")
        exit(1)
    out_dir = os.path.join('/home/yiwei-guest/code/waymo/output', os.path.basename(os.path.normpath(target_dir)))
    os.makedirs(out_dir, exist_ok=True)

    # 初始化深度图插值用的卷积核
    kernel_shape = [17, 7]
    kernel_y = [2**i for i in range(int(kernel_shape[0] / 2) + 1)]
    kernel_y = kernel_y + kernel_y[-2::-1]
    kernel_y = np.array(kernel_y).reshape(-1, 1)
    kernel_x = [2**i for i in range(int(kernel_shape[1] / 2) + 1)]
    kernel_x = kernel_x + kernel_x[-2::-1]
    kernel_x = np.array(kernel_x).reshape(1, -1)
    kernel = kernel_y @ kernel_x

    file_names = os.listdir(target_dir)
    for file_name in file_names:
        TFRecord_PATH = os.path.join(target_dir, file_name)
        print(f"Loading {TFRecord_PATH}")
        dataset = tf.data.TFRecordDataset(TFRecord_PATH, compression_type='')
        print(f'successfully loaded {TFRecord_PATH}')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # 仅保留包含分割标签的帧
            if frame.images[0].camera_segmentation_label.panoptic_label and frame.lasers[0].ri_return1.segmentation_label_compressed:
                print(f"find labels in {TFRecord_PATH}")
                frame_dic = {}

                (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

                # 相机内参
                intrinsic = frame.context.camera_calibrations[open_dataset.CameraName.FRONT].intrinsic
                K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
                D = np.array([intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]])
                frame_dic['K'] = K
                frame_dic['D'] = D

                # 距离图像
                top_range_image = tf.reshape(range_images[1][0].data, range_images[1][0].shape.dims)
                frame_dic['range_image'] = top_range_image[...,0].numpy()

                # 距离图像的语义标签
                semseg_label_image = segmentation_labels[open_dataset.LaserName.TOP][0]
                semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
                semseg_label_image_tensor = tf.reshape(
                    semseg_label_image_tensor, semseg_label_image.shape.dims)
                instance_id_image = semseg_label_image_tensor[...,0] 
                semantic_class_image = semseg_label_image_tensor[...,1].numpy()
                semantic_class_image = lookup_table_3d[semantic_class_image]
                frame_dic['range_image_semantic'] = semantic_class_image

                # RGB图像
                image = next(img for img in frame.images if img.name == open_dataset.CameraName.FRONT)
                image_array = tf.image.decode_jpeg(image.image).numpy()
                frame_dic['image'] = image_array

                # RGB图像的语义标签
                panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
                    image.camera_segmentation_label
                )
                semantic_label, _ = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                    panoptic_label,
                    image.camera_segmentation_label.panoptic_label_divisor
                )
                semantic_label = lookup_table_2d[semantic_label]
                frame_dic['image_semantic'] = semantic_label

                # 位置
                frame_dic['pose'] = np.array(frame.pose.transform).reshape(4, 4)

                points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
                points_all = np.concatenate(points, axis=0)
                cp_points_all = np.concatenate(cp_points, axis=0)
                images = sorted(frame.images, key=lambda i:i.name)

                # The distance between lidar points and vehicle frame origin.
                points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
                cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
                
                mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)  # 选出图片对应的投影点云
                cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
                points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
                projected_points_all_from_raw_data = tf.concat([cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

                # 只取图片的下半部分
                image_array = image_array[580:, :, :]
                projected_points_all_from_raw_data[:, 1] -= 580
                frame_dic['image_dp_slice'] = image_array

                height, width = image_array.shape[:-1]
                depth_image = np.zeros(image_array.shape[:-1], dtype=np.float32)
                for i in range(len(projected_points_all_from_raw_data)):
                    if not ((0 <= projected_points_all_from_raw_data[i][0] <= width - 1)
                            and (0 <= projected_points_all_from_raw_data[i][1] <= height - 1)):
                        continue
                    u_ = int(round(projected_points_all_from_raw_data[i][0]))
                    v_ = int(round(projected_points_all_from_raw_data[i][1]))
                    depth_image[v_, u_] = projected_points_all_from_raw_data[i][2]

                bin_depth = depth_image != 0
                weight_sum = cv2.filter2D(bin_depth.astype(np.float32), -1, kernel)
                depth_image = cv2.filter2D(depth_image, -1, kernel)
                depth_image = depth_image / weight_sum
                frame_dic['depth_image'] = depth_image
                
                save_name = os.path.basename(os.path.normpath(target_dir)) + '_' + str(count).zfill(4) + '.npz'
                np.savez(os.path.join(out_dir, save_name), **frame_dic)
                print(f"frame {count} loaded")
                count += 1

                # # 可视化,检查标签映射是否正确
                # semantic_rgb = camera_segmentation_utils.semantic_label_to_rgb(semantic_label)
                # bar = np.zeros((semantic_rgb.shape[0], semantic_rgb.shape[1] + 80, 1), dtype=np.uint8)
                # bar[:semantic_rgb.shape[0], :semantic_rgb.shape[1], :] = semantic_label
                # part = semantic_rgb.shape[0] / 9
                # for i in range(9):
                #     bar[int(part * i):int(part * (i + 1)), semantic_rgb.shape[1]:, 0] = i
                # semantic_rgb = camera_segmentation_utils.semantic_label_to_rgb(bar)

                # plt.figure()

                # plt.subplot(1, 1, 1)
                # plt.imshow(semantic_rgb)
                # plt.title("Semantic Segmentation")
                # plt.axis('off')
                # plt.show()
                
                # bar_image = np.zeros((semantic_class_image.shape[0] + 30, semantic_class_image.shape[1], 1), dtype=np.uint8)
                # bar_image[:semantic_class_image.shape[0], :semantic_class_image.shape[1], 0] = semantic_class_image
                # part = semantic_class_image.shape[1] / 9
                # for i in range(9):
                #     bar_image[semantic_class_image.shape[0]:, int(part * i):int(part * (i + 1)), 0] = i
                # plt.figure()
                # plt.subplot(2, 1, 1)
                # plt.imshow(bar_image, vmin=0, vmax=8, cmap='Paired')
                # plt.title('range image')
                # plt.grid(False)
                # plt.axis('off')
                # plt.subplot(2, 1, 2)
                # plt.imshow(bar_image[semantic_class_image.shape[0]:, :, :], vmin=0, vmax=8, cmap='Paired')
                # plt.title('bar')
                # plt.grid(False)
                # plt.axis('off')
                # plt.show()

    print(f"{count} frames loaded")