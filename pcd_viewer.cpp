/* ******************************************************************************************************************************************
 * 3D Point Cloud Mapping Algorithm (2022)																									*
 * @author https://github.com/vyanev1																										*
 * ******************************************************************************************************************************************/

#include <random>
#include <thread>

#include <pcl/console/time.h> // TicToc
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>

// ------------------------------------------------------------ Global Variables ------------------------------------------------------------

typedef pcl::PointXYZRGBA PointT;

pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
pcl::PassThrough<PointT> pass;
pcl::console::TicToc time_it;

static constexpr float SPHERE_RADIUS = 0.03f;

float low_limit = 0.7f;
float high_limit = 1;
bool changed_limits = false;


// ------------------------------------------------------------ Viewer Callbacks ------------------------------------------------------------

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
	viewer.initCameraParameters();
}

void viewerPsycho(pcl::visualization::PCLVisualizer& viewer)
{
	//std::stringstream ss;
	//ss << "Low limit: " << low_limit << "; High limit: " << high_limit;
	//viewer.removeShape("text", 0);
	//viewer.addText(ss.str(), 200, 300, "text", 0);
	if (changed_limits) {
		pass.setFilterLimits(low_limit, high_limit);
		pass.filter(*cloud);
		viewer.updatePointCloud(cloud);
		changed_limits = false;
	}
}


// ---------------------------------------------------------------- Filters -----------------------------------------------------------------

void applyPassThroughFilter(const pcl::PointCloud<PointT>::Ptr& in_cloud)
{
	cout << "[INFO] Applying PassThrough filter..." << endl;
	pass.setInputCloud(in_cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(low_limit, high_limit);
	pass.setFilterLimitsNegative(false);
	pass.filter(*in_cloud);
}

void applyRGBColorFilter(const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<PointT>::Ptr& cloud_output)
{
	cout << "[INFO] Applying RGB filter for calibration object" << endl;
	const pcl::ConditionAnd<PointT>::Ptr color_condition(new pcl::ConditionAnd<PointT>());
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("r", pcl::ComparisonOps::LT, 130));
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("g", pcl::ComparisonOps::LT, 130));
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("b", pcl::ComparisonOps::GT, 90));

	// Build the filter
	pcl::ConditionalRemoval<PointT> color_filter;
	color_filter.setInputCloud(cloud);
	color_filter.setCondition(color_condition);
	color_filter.filter(*cloud_output);
}


// ----------------------------------------------------------- Normal Estimation ------------------------------------------------------------

void computeNormals(const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
	cout << "[INFO] Computing normals..." << endl;
	const pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.01);
	ne.setInputCloud(cloud);
	ne.useSensorOriginAsViewPoint();
	ne.compute(*normals);
}


// ------------------------------------------------------------ Helper Methods --------------------------------------------------------------

void concatenateFields(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud1_in, const pcl::PointCloud<pcl::Normal>& cloud2_in, pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_out)
{
	if (cloud1_in.points.size() != cloud2_in.points.size())
	{
		PCL_ERROR("[pcl::concatenateFields] The number of points in the two input datasets differs!\n");
		return;
	}

	// Resize the output dataset
	cloud_out.points.resize(cloud1_in.points.size());
	cloud_out.header = cloud1_in.header;
	cloud_out.width = cloud1_in.width;
	cloud_out.height = cloud1_in.height;
	if (!cloud1_in.is_dense || !cloud2_in.is_dense)
		cloud_out.is_dense = false;
	else
		cloud_out.is_dense = true;

	// Iterate over each point
	for (size_t i = 0; i < cloud_out.points.size(); ++i)
	{
		const pcl::PointXYZRGBA& x1 = cloud1_in[i];
		const pcl::Normal& x2 = cloud2_in[i];
		pcl::PointXYZRGBNormal& y = cloud_out[i];
		memcpy(y.data, x1.data, sizeof(x1.data));			// coordinates
		memcpy(y.data_n, x2.data_n, sizeof(x2.data_n));	// normal
		y.rgba = x1.rgba;										// color
		y.curvature = x2.curvature;								// curvature
	}
}

template <typename PointType>
PointType getNearestPoint(pcl::PointCloud<PointType>& cloud)
{
	cout << "[INFO] Finding nearest point..." << endl;

	float min_distance = FLT_MAX;
	PointType min_distance_point;
	for (const auto& point : cloud.points)
	{
		if (point.z > low_limit && point.z < min_distance)
		{
			min_distance = point.z;
			min_distance_point = point;
		}
	}
	cout << "coords:  " << min_distance_point.x << " " << min_distance_point.y << " " << min_distance_point.z << endl;

	if constexpr (std::is_same_v<PointType, pcl::PointXYZRGBNormal>)
		cout << "normals: " << min_distance_point.normal_x << " " << min_distance_point.normal_y << " " << min_distance_point.normal_z << endl;

	return min_distance_point;
}

void print4x4Matrix(const Eigen::Matrix4d& matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}


// ----------------------------------------------------------- Sphere Generation ------------------------------------------------------------

PointT randomUnitSpherePoint()
{
	PointT point;
	// cylindrical coordinates
	const double rxy = sqrt(1 - pow(Eigen::internal::random(0., 1.), (2. / 3.)));
	const double phi = Eigen::internal::random(0., 2 * M_PI);
	const double zAbsMax = sqrt(1 - rxy * rxy);
	point.z = static_cast<float>(Eigen::internal::random(-zAbsMax, zAbsMax));
	// cartesian coordinates
	point.x = static_cast<float>(rxy * cos(phi));
	point.y = static_cast<float>(rxy * sin(phi));
	return point;
}

PointT randomSpherePoint(const PointT center, const float r)
{
	PointT point = randomUnitSpherePoint();
	point.x = center.x + r * point.x;
	point.y = center.y + r * point.y;
	point.z = center.z + r * point.z;
	return point;
}

void createSphere(const pcl::PointXYZRGBNormal& input_point, float r, std::string::value_type color, const pcl::PointCloud<PointT>::Ptr& sphere)
{
	cout << "[INFO] Artificially generating spherical volume of points..." << endl;
	PointT center;
	center.x = input_point.x - r * input_point.normal_x;
	center.y = input_point.y - r * input_point.normal_y;
	center.z = input_point.z - r * input_point.normal_z;

	int index = 0;
	for (size_t i = 0; i < sphere->width; i++)
	{
		for (size_t j = 0; j < sphere->height; j++)
		{
			sphere->points[index] = randomSpherePoint(center, r);
			switch (color)
			{
				case 'R': sphere->points[index].r = 255; break;
				case 'G': sphere->points[index].g = 255; break;
				case 'B': sphere->points[index].b = 255; break;
				default:;
			}
			index++;
		}
	}
}

void createSphereSurface(const pcl::PointXYZRGBNormal input_point, const float radius, const char color, const pcl::PointCloud<PointT>::Ptr& sphere_cloud)
{
	cout << "[INFO] Drawing sphere surface..." << endl;
	int index = 0;
	for (size_t m = 0; m < sphere_cloud->height; m++)
	{
		for (size_t n = 0; n < sphere_cloud->width - 1; n++)
		{
			const double x = std::sin(M_PI * static_cast<double>(m) / sphere_cloud->height) * std::cos(2 * M_PI * static_cast<double>(n) / sphere_cloud->width);
			const double y = std::sin(M_PI * static_cast<double>(m) / sphere_cloud->height) * std::sin(2 * M_PI * static_cast<double>(n) / sphere_cloud->width);
			const double z = std::cos(M_PI * static_cast<double>(m) / sphere_cloud->height);

			sphere_cloud->points[index].x = (input_point.x - radius * input_point.normal_x) + (static_cast<float>(x) * radius);
			sphere_cloud->points[index].y = (input_point.y - radius * input_point.normal_y) + (static_cast<float>(y) * radius);
			sphere_cloud->points[index].z = (input_point.z - radius * input_point.normal_z) + (static_cast<float>(z) * radius);

			switch (color)
			{
			case 'R': sphere_cloud->points[index].r = 255; break;
			case 'G': sphere_cloud->points[index].g = 255; break;
			case 'B': sphere_cloud->points[index].b = 255; break;
			default:;
			}

			index++;
		}
	}
}

// ----------------------------------------------------------- Initial Guess Estimation -----------------------------------------------------------
Eigen::Matrix4d estimateRotation(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	double theta = 0;  // The angle of rotation in radians

	const PointT nearestPoint = getNearestPoint<PointT>(*cloud);

	if (nearestPoint.r == 255) theta = M_PI / 3;		// 60 degrees
	if (nearestPoint.g == 255) theta = M_PI;			// 180 degrees
	if (nearestPoint.b == 255) theta = M_PI * 5 / 3;	// 300 degrees

	transformation_matrix(0, 0) = cos(theta);
	transformation_matrix(0, 1) = -sin(theta);
	transformation_matrix(1, 0) = sin(theta);
	transformation_matrix(1, 1) = cos(theta);

	// A translation on Z axis (0.4 meters)
	//transformation_matrix(2, 3) = 0.4;

	return transformation_matrix;
}

Eigen::Matrix4d estimateInitialGuess(const pcl::PointCloud<PointT>::Ptr source, pcl::PointCloud<PointT>::Ptr target)
{
	Eigen::Matrix4d rotation_matrix_src = estimateRotation(source);
	Eigen::Matrix4d rotation_matrix_tgt = estimateRotation(target);

	Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
	result(0, 0) = rotation_matrix_tgt(0, 0) - rotation_matrix_src(0, 0);
	result(0, 1) = rotation_matrix_tgt(0, 1) - rotation_matrix_src(0, 1);
	result(1, 0) = rotation_matrix_tgt(1, 0) - rotation_matrix_src(1, 0);
	result(1, 1) = rotation_matrix_tgt(1, 1) - rotation_matrix_src(1, 1);

	return result;
}

// ----------------------------------------------------------- Main Function -----------------------------------------------------------
int main()
{
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<PointT>::Ptr>> clouds_to_combine;
	const std::string colors = "RGB";

	for (int camera_index = 1; camera_index <= 2; camera_index++)
	{
		pcl::PointCloud<PointT>::Ptr calibration_cloud(new pcl::PointCloud<PointT>);

		for (int color_index = 0; color_index < colors.size(); color_index++)
		{
			// E.g. "camera_1_R.pcd", "camera_1_G.pcd", "camera_1_B.pcd"
			std::string filename = "camera_" + std::to_string(camera_index) + "_" + colors[color_index] + ".pcd";

			cout << "[INFO] Fetching " << filename << "..." << endl;
			const pcl::PointCloud<PointT>::Ptr curr_cloud(new pcl::PointCloud<PointT>);
			pcl::io::loadPCDFile(filename, *curr_cloud);

			cout << "[INFO] Inverting X & Y coordinates..." << endl;
			for (auto& point : curr_cloud->points)
			{
				point.x *= -1;
				point.y *= -1;
			}

			pcl::PointCloud<PointT>::Ptr cloud_color_filtered(new pcl::PointCloud<PointT>);
			applyPassThroughFilter(curr_cloud);
			applyRGBColorFilter(curr_cloud, cloud_color_filtered);

			const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
			const pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
			computeNormals(cloud_color_filtered, normals);
			concatenateFields(*cloud_color_filtered, *normals, *cloud_with_normals);
			
			const pcl::PointXYZRGBNormal nearest_point = getNearestPoint<pcl::PointXYZRGBNormal>(*cloud_with_normals);

			const pcl::PointCloud<PointT>::Ptr sphere_cloud(new pcl::PointCloud<PointT>(50, 50));
			createSphere(nearest_point, SPHERE_RADIUS, colors[color_index], sphere_cloud);
			*curr_cloud += *sphere_cloud;

			cout << "[INFO] Adding " << filename << " to PointCloud object of current camera." << endl;
			*calibration_cloud += *sphere_cloud;
		}
		cout << "[INFO] Adding pre-processed view of camera " << camera_index << " to list of clouds to align." << endl;
		clouds_to_combine.push_back(calibration_cloud);
	}

	// Initial Guess
	Eigen::Matrix4d initial_guess = estimateInitialGuess(clouds_to_combine.at(0), clouds_to_combine.at(1));

	// The Iterative Closest Point algorithm
	time_it.tic();
	pcl::PointCloud<PointT>::Ptr combined_cloud(new pcl::PointCloud<PointT>(*clouds_to_combine.at(1)));
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(50);
	icp.setInputSource(clouds_to_combine.at(0));
	icp.setInputTarget(clouds_to_combine.at(1));
	icp.align(*combined_cloud, initial_guess.cast<float>());
	std::cout << "Applied " << 50 << " ICP iteration(s) in " << time_it.toc() << " ms" << std::endl;

	icp.setMaximumIterations(1); // We set this variable to 1 for the next time we will call .align() function

	if (icp.hasConverged()) {
		std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
		std::cout << "\nICP transformation " << 50 << " : cloud_icp -> cloud_in" << std::endl;
		Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
		print4x4Matrix(transformation_matrix);
	} else {
		PCL_ERROR("\nICP has not converged.\n");
		return (-1);
	}

	cout << "[INFO] Opening CloudViewer..." << endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(combined_cloud);
	viewer.runOnVisualizationThreadOnce(viewerOneOff); // This will only get called once
	viewer.runOnVisualizationThread(viewerPsycho);     // This will get called once per visualization iteration

	while (!viewer.wasStopped())
	{
		if (getchar() != 0)
		{
			low_limit += 0.1f;
			high_limit += 0.1f;
			changed_limits = true;
		}
	}

	return 0;
}

//int main()
//{
//	const std::string colors = "RGB";
//	for (int camera_index = 1; camera_index <= 2; camera_index++)
//	{
//		for (int color_index = 0; color_index < colors.size(); color_index++)
//		{
//			// E.g. "camera_1_R.pcd", "camera_1_G.pcd", "camera_1_B.pcd"
//			std::string filename = "camera_" + std::to_string(camera_index) + "_" + colors[color_index] + ".pcd";
//
//			cout << "[INFO] Fetching " << filename << "..." << endl;
//			const pcl::PointCloud<PointT>::Ptr curr_cloud(new pcl::PointCloud<PointT>);
//			pcl::io::loadPCDFile(filename, *curr_cloud);
//
//			cout << "[INFO] Removing NaN points..." << endl;
//			std::vector<int> indices;
//			removeNaNFromPointCloud(*curr_cloud, *curr_cloud, indices);
//
//			cout << "[INFO] Removing outliers..." << endl;
//			pcl::StatisticalOutlierRemoval<PointT> sor;
//			sor.setInputCloud(curr_cloud);
//			sor.setMeanK(50);
//			sor.setStddevMulThresh(1.0);
//			sor.filter(*curr_cloud);
//
//			cout << "[INFO] Saving " << filename << " file of current camera." << endl;
//			pcl::io::savePCDFile(filename, *curr_cloud);
//		}
//	}
//	return 0;
//}
