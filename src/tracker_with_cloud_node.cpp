/**
 * ultralytics_ros
 * Copyright (C) 2023-2024  Alpaca-zip
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tracker_with_cloud_node/tracker_with_cloud_node.h"
#include "dbscan_kdtree/DBSCAN_kdtree.h"

TrackerWithCloudNode::TrackerWithCloudNode() : pnh_("~")
{
  pnh_.param<std::string>("camera_info_topic", camera_info_topic_, "camera_info");
  pnh_.param<std::string>("lidar_topic", lidar_topic_, "points_raw");
  pnh_.param<std::string>("yolo_result_topic", yolo_result_topic_, "yolo_result");
  pnh_.param<std::string>("yolo_3d_result_topic", yolo_3d_result_topic_, "yolo_3d_result");
  pnh_.param<float>("cluster_tolerance", cluster_tolerance_, 0.5);
  pnh_.param<int>("min_cluster_size", min_cluster_size_, 100);
  pnh_.param<int>("max_cluster_size", max_cluster_size_, 25000);

  detection_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detection_cloud", 1);
  detection3d_pub_ = nh_.advertise<vision_msgs::Detection3DArray>(yolo_3d_result_topic_, 1);
  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("detection_marker", 1);
  camera_info_sub_.subscribe(nh_, camera_info_topic_, 10);
  lidar_sub_.subscribe(nh_, lidar_topic_, 1);
  yolo_result_sub_.subscribe(nh_, yolo_result_topic_, 10);
  sync_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(10);
  sync_->connectInput(camera_info_sub_, lidar_sub_, yolo_result_sub_);
  sync_->registerCallback(boost::bind(&TrackerWithCloudNode::syncCallback, this, _1, _2, _3));

  tf_buffer_.reset(new tf2_ros::Buffer(ros::Duration(2.0), true));
  tf_listener_.reset(new tf2_ros::TransformListener(*tf_buffer_));
}

void TrackerWithCloudNode::syncCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
                                        const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                        const ultralytics_ros::YoloResultConstPtr& yolo_result_msg)
{
  ros::Time current_call_time = ros::Time::now();
  ros::Duration callback_interval = current_call_time - last_call_time_;
  last_call_time_ = current_call_time;
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  downsampled_cloud = downsampleCloudMsg(cloud_msg);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud;
  transformed_cloud = cloud2TransformedCloud(downsampled_cloud, cloud_msg->header.frame_id, cam_model_.tfFrame(),
                                             cloud_msg->header.stamp);
  vision_msgs::Detection3DArray detections3d_msg;
  sensor_msgs::PointCloud2 detection_cloud_msg;
  visualization_msgs::MarkerArray marker_array_msg;

  cam_model_.fromCameraInfo(camera_info_msg);
  projectCloud(transformed_cloud, yolo_result_msg, cloud_msg->header, detections3d_msg, detection_cloud_msg);
  marker_array_msg = createMarkerArray(detections3d_msg, callback_interval.toSec());

  detection3d_pub_.publish(detections3d_msg);
  detection_cloud_pub_.publish(detection_cloud_msg);
  marker_pub_.publish(marker_array_msg);
}

static void getMinMax3D(pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                        pcl::Indices& indices, Eigen::Vector4f& min_pt,
                        Eigen::Vector4f& max_pt) {
  float min_x = cloud->points[0].x, min_y = cloud->points[0].y,
        min_z = cloud->points[0].z, max_x = cloud->points[0].x,
        max_y = cloud->points[0].y, max_z = cloud->points[0].z;

  for (auto const &i : indices) {
    if (cloud->points[i].x <= min_x)
      min_x = cloud->points[i].x;
    else if (cloud->points[i].y <= min_y)
      min_y = cloud->points[i].y;
    else if (cloud->points[i].z <= min_z)
      min_z = cloud->points[i].z;
    else if (cloud->points[i].x >= max_x)
      max_x = cloud->points[i].x;
    else if (cloud->points[i].y >= max_y)
      max_y = cloud->points[i].y;
    else if (cloud->points[i].z >= max_z)
      max_z = cloud->points[i].z;
  }

  min_pt = Eigen::Vector4f(min_x, min_y, min_z, 0.0);
  max_pt = Eigen::Vector4f(max_x, max_y, max_z, 0.0);
}

void TrackerWithCloudNode::projectCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                        const ultralytics_ros::YoloResultConstPtr& yolo_result_msg,
                                        const std_msgs::Header& header, vision_msgs::Detection3DArray& detections3d_msg,
                                        sensor_msgs::PointCloud2& combine_detection_cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ> combine_detection_cloud;
  detections3d_msg.header = header;
  detections3d_msg.header.stamp = yolo_result_msg->header.stamp;

  auto detections = yolo_result_msg->detections;

  std::sort(detections.detections.begin(), detections.detections.end(), [](auto const& lhs, auto const& rhs){
     return lhs.bbox.center.y > rhs.bbox.center.y;
  });

  auto clusters = euclideanClusterExtraction(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto const& detection : detections.detections)
  {
    detection_cloud_raw->resize(0);
    processPointsWithBbox(cloud, clusters, detection, detection_cloud_raw);

    if (detection_cloud_raw->empty() == false)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud =
        cloud2TransformedCloud(detection_cloud_raw, this->cam_model_.tfFrame(), header.frame_id, header.stamp);
      createBoundingBox(detections3d_msg, detection_cloud, detection.results);
      combine_detection_cloud += *detection_cloud;
    }
  }

  if (clusters.empty() == false)
  {
    for (auto const& cluster : clusters)
    {
      Eigen::Vector4f min_pt, max_pt, center_pt;
      getMinMax3D(*cloud, cluster.indices, min_pt, max_pt);
      center_pt = (min_pt + max_pt) / 2;
      createUnknownBoundingBox(detections3d_msg, center_pt, this->cam_model_.tfFrame(), header.frame_id, header.stamp);
    }
  }

  pcl::toROSMsg(combine_detection_cloud, combine_detection_cloud_msg);
  combine_detection_cloud_msg.header = header;
}

void TrackerWithCloudNode::processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                 std::vector<pcl::PointIndices>& clusters,
                                                 const vision_msgs::Detection2D& detection,
                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud_raw)
{
  int32_t n_cpus = omp_get_max_threads();

  float closest_distance = std::numeric_limits<float>::max();
  auto closest_indice = clusters.end();

  std::vector<typeof(closest_distance)> closest_distances(n_cpus, closest_distance);
  std::vector<typeof(closest_indice)> closest_indices(n_cpus, closest_indice);

#pragma omp parallel for
  for (auto cluster = clusters.begin(); cluster < clusters.end(); ++cluster)
  {
    int32_t thread_num = omp_get_thread_num();
    Eigen::Vector4f min_pt, max_pt, center_pt, size_pt;
    getMinMax3D(*cloud, cluster->indices, min_pt, max_pt);
    center_pt = (min_pt + max_pt) / 2;
    size_pt = max_pt - min_pt;

    auto const distance = pow(center_pt[0], 2) + pow(center_pt[1], 2) + pow(center_pt[2], 2);
    cv::Point3d const pt_cv(center_pt[0], center_pt[1], center_pt[2]);
    auto const uv = cam_model_.project3dToPixel(pt_cv);

    auto const lu_uv = cv::Point2d(detection.bbox.center.x - detection.bbox.size_x / 2,
                                   detection.bbox.center.y - detection.bbox.size_y / 2);
    auto const rd_uv = cv::Point2d(detection.bbox.center.x + detection.bbox.size_x / 2,
                                   detection.bbox.center.y + detection.bbox.size_y / 2);
    int const offset = 15;

    if (uv.x + offset >= lu_uv.x && uv.y + offset >= lu_uv.y &&
        uv.x - offset <= rd_uv.x && uv.y - offset <= rd_uv.y && distance < closest_distances[thread_num])
    {
      closest_distances[thread_num] = distance;
      closest_indices[thread_num] = cluster;
    }
  }

  for (uint32_t i = 0; i < n_cpus; ++i) {
    if (closest_distances[i] < closest_distance) {
      closest_distance = closest_distances[i];
      closest_indice = closest_indices[i];
    }
  }

  if (closest_distance < 100 && closest_indice < clusters.end()) {
    for (auto const& indice : closest_indice->indices) {
      detection_cloud_raw->points.push_back(cloud->points[indice]);
    }
    *closest_indice = clusters[clusters.size() - 1];
    clusters.resize(clusters.size() - 1);
  }
}

void TrackerWithCloudNode::processPointsWithMask(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                 const sensor_msgs::Image& mask,
                                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud_raw)
{
  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(mask, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  for (const auto& point : cloud->points)
  {
    cv::Point3d pt_cv(point.x, point.y, point.z);
    cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);

    if (point.z > 0 && uv.x >= 0 && uv.x < cv_ptr->image.cols && uv.y >= 0 && uv.y < cv_ptr->image.rows)
    {
      if (cv_ptr->image.at<uchar>(cv::Point(uv.x, uv.y)) > 0)
      {
        detection_cloud_raw->points.push_back(point);
      }
    }
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::downsampleCloudMsg(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud_msg, *cloud);

  float const max_squared_distance = 1000;
  for (uint32_t i = 0; i < cloud->points.size(); ++i)
  {
    auto const& point = cloud->points[i];
    float const squared_distance = std::pow(point.x, 2) + std::pow(point.y, 2) + std::pow(point.z, 2);

    if (squared_distance > max_squared_distance || point.x < 1 || point.z > -0.5 || point.z < -1.0)
    {
      cloud->points[i] = cloud->points[cloud->points.size() - 1];
      cloud->points.resize(cloud->points.size() - 1);
      --i;
    }
  }

  ROS_INFO("Downsampled size: %ld", cloud->points.size());

  return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::cloud2TransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                             const std::string& source_frame, const std::string& target_frame,
                                             const ros::Time& stamp)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  try
  {
    geometry_msgs::TransformStamped tf = tf_buffer_->lookupTransform(target_frame, source_frame, stamp);
    pcl_ros::transformPointCloud(*cloud, *transformed_cloud, tf.transform);
  }
  catch (tf2::TransformException& e)
  {
    ROS_WARN("%s", e.what());
  }

  return transformed_cloud;
}

std::vector<pcl::PointIndices>
TrackerWithCloudNode::euclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  std::vector<pcl::PointIndices> cluster_indices;
  DBSCANKdtreeCluster<pcl::PointXYZ> dc;

  tree->setInputCloud(cloud);

  dc.setCorePointMinPts(this->min_cluster_size_);
  dc.setMinClusterSize(this->min_cluster_size_);
  dc.setMaxClusterSize(this->max_cluster_size_);
  dc.setClusterTolerance(this->cluster_tolerance_);
  dc.setSearchMethod(tree);
  dc.setInputCloud(cloud);
  dc.extract(cluster_indices);

  ROS_INFO("Found %ld clusters", cluster_indices.size());

  return cluster_indices;
}

void TrackerWithCloudNode::createBoundingBox(
    vision_msgs::Detection3DArray& detections3d_msg, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<vision_msgs::ObjectHypothesisWithPose>& detections_results)
{
  vision_msgs::Detection3D detection3d;
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointXYZ min_pt, max_pt;
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  double theta = -atan2(centroid[1], sqrt(pow(centroid[0], 2) + pow(centroid[2], 2)));

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));
  pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

  pcl::getMinMax3D(*transformed_cloud, min_pt, max_pt);
  Eigen::Vector4f transformed_bbox_center =
      Eigen::Vector4f((min_pt.x + max_pt.x) / 2, (min_pt.y + max_pt.y) / 2, (min_pt.z + max_pt.z) / 2, 1);
  Eigen::Vector4f bbox_center = transform.inverse() * transformed_bbox_center;
  Eigen::Quaternionf q(transform.inverse().rotation());

  detection3d.bbox.center.position.x = bbox_center[0];
  detection3d.bbox.center.position.y = bbox_center[1];
  detection3d.bbox.center.position.z = bbox_center[2];
  detection3d.bbox.center.orientation.x = q.x();
  detection3d.bbox.center.orientation.y = q.y();
  detection3d.bbox.center.orientation.z = q.z();
  detection3d.bbox.center.orientation.w = q.w();
  detection3d.bbox.size.x = max_pt.x - min_pt.x;
  detection3d.bbox.size.y = max_pt.y - min_pt.y;
  detection3d.bbox.size.z = max_pt.z - min_pt.z;
  detection3d.results = detections_results;
  detections3d_msg.detections.push_back(detection3d);
}

void TrackerWithCloudNode::createUnknownBoundingBox(
  vision_msgs::Detection3DArray& detections3d_msg, Eigen::Vector4f bbox_center,
  std::string const& source_frame, std::string const& target_frame, ros::Time const& stamp
)
{
  try
  {
    geometry_msgs::TransformStamped tf =
      this->tf_buffer_->lookupTransform(target_frame, source_frame, stamp);
    Eigen::Matrix4f transform;
    pcl_ros::transformAsMatrix(tf.transform, transform);
    bbox_center = transform * bbox_center;
  }
  catch (tf2::TransformException& e)
  {
    ROS_WARN("%s", e.what());  
  }
  
  vision_msgs::Detection3D detection3d;
  detection3d.bbox.center.position.x = bbox_center[0];
  detection3d.bbox.center.position.y = bbox_center[1];
  detection3d.bbox.center.position.z = bbox_center[2];
  detection3d.bbox.size.x = 0.5;
  detection3d.bbox.size.y = 0.5;
  detection3d.bbox.size.z = 0.5;

  vision_msgs::ObjectHypothesisWithPose pose;
  pose.id = 3;
  pose.score = 1.0;
  detection3d.results.push_back(pose);
  detections3d_msg.detections.push_back(detection3d);
}

visualization_msgs::MarkerArray
TrackerWithCloudNode::createMarkerArray(const vision_msgs::Detection3DArray& detections3d_msg, const double& duration)
{
  visualization_msgs::MarkerArray marker_array;
  for (size_t i = 0; i < detections3d_msg.detections.size(); i++)
  {
    if (std::isfinite(detections3d_msg.detections[i].bbox.size.x) &&
        std::isfinite(detections3d_msg.detections[i].bbox.size.y) &&
        std::isfinite(detections3d_msg.detections[i].bbox.size.z))
    {
      visualization_msgs::Marker marker;
      marker.header = detections3d_msg.header;
      marker.ns = "detection";
      marker.id = i;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose = detections3d_msg.detections[i].bbox.center;
      marker.scale.x = detections3d_msg.detections[i].bbox.size.x;
      marker.scale.y = detections3d_msg.detections[i].bbox.size.y;
      marker.scale.z = detections3d_msg.detections[i].bbox.size.z;
      marker.color.r = detections3d_msg.detections[i].results[0].id == 2 ||
                       detections3d_msg.detections[i].results[0].id == 1;
      marker.color.g = detections3d_msg.detections[i].results[0].id == 1;
      marker.color.b = detections3d_msg.detections[i].results[0].id == 0;
      marker.color.a = 0.5;
      marker.lifetime = ros::Duration(duration);
      marker_array.markers.push_back(marker);
    }
  }
  return marker_array;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tracker_with_cloud_node");
  TrackerWithCloudNode node;
  ros::spin();
}
