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

pcl::visualization::PCLVisualizer* viewer;
int vp_1, vp_2, vp_3; // Viewports

float low_limit = 0.7f;
float high_limit = 1;
bool changed_limits = false;

bool next_iteration = false;
int iterations = 25;

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

void showCloudsLeft(const pcl::PointCloud<PointT>::Ptr& cloud_target, const pcl::PointCloud<PointT>::Ptr& cloud_source)
{
	viewer->addPointCloud(cloud_target, "vp1_target", vp_1);
	viewer->addPointCloud(cloud_source, "vp1_source", vp_2);
	viewer->addText("Target point cloud", 10, 15, 16, 255, 255, 255, "icp_info_1", vp_1);
	viewer->addText("Source point cloud", 10, 15, 16, 255, 255, 255, "icp_info_2", vp_2);
	viewer->spinOnce();
	std::cout << "Completed Left ViewPort" << std::endl;
}

void showCloudsRight(const pcl::PointCloud<PointT>::Ptr &cloud_target, const pcl::PointCloud<PointT>::Ptr& cloud_source)
{
	viewer->addPointCloud(cloud_target, "target", vp_3);
	viewer->addPointCloud(cloud_source, "source", vp_3);
	viewer->spinOnce();
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
		next_iteration = true;
}

void print4x4Matrix(const Eigen::Matrix4f& matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
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
PointType getNearestPoint(pcl::PointCloud<PointType>& cloud, const Eigen::Vector3i &exclude_color = Eigen::Vector3i(-1, -1, -1))
{
	exclude_color(0) == -1 ? cout << "[INFO] Finding nearest point..." << endl : cout << "[INFO] Finding second nearest point..." << endl;

	float min_distance = FLT_MAX;
	PointType min_distance_point;
	for (const auto& point : cloud.points)
	{
		const float distance_from_origin = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
		if (distance_from_origin < min_distance)
		{
			if (!exclude_color.isApprox(point.getRGBVector3i())) {
				min_distance = abs(point.z);
				min_distance_point = point;
			}
		}
	}
	cout << "coords:  " << min_distance_point.x << " " << min_distance_point.y << " " << min_distance_point.z << endl;

	if constexpr (std::is_same_v<PointType, pcl::PointXYZRGBNormal>)
		cout << "normals: " << min_distance_point.normal_x << " " << min_distance_point.normal_y << " " << min_distance_point.normal_z << endl;

	return min_distance_point;
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
double translate(const double value, const double left_min, const double left_max, const double right_min, const double right_max)
{
	const double left_span = left_max - left_min;
	const double right_span = right_max - right_min;
	const double value_scaled = (value - left_min) / left_span; // Convert the left range into a 0 to 1 range
	return right_min + value_scaled * right_span;
}

float estimateRotation(const pcl::PointCloud<PointT>::Ptr& cloud)
{
	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();

	const PointT nearest_point = getNearestPoint<PointT>(*cloud);
	const PointT second_nearest_point = getNearestPoint<PointT>(*cloud, nearest_point.getRGBVector3i());
	const double d1 = sqrt(pow(nearest_point.x, 2) + pow(nearest_point.y, 2) + pow(nearest_point.z, 2));
	const double d2 = sqrt(pow(second_nearest_point.x, 2) + pow(second_nearest_point.y, 2) + pow(second_nearest_point.z, 2));
	const double min_dist = d1;
	const double max_dist = d1 + 0.1;

	cout << "d1: " << d1 << endl;
	cout << "d2: " << d2 << endl;
	cout << "min_dist: " << min_dist << endl;
	cout << "max_dist: " << max_dist << endl;

	float theta = 0; // The angle of rotation in radians

	if (nearest_point.r == 255) {
		cout << "[INFO] Nearest point is red." << endl;
		theta = second_nearest_point.b == 255
			? - static_cast<float>(translate(d2, min_dist, max_dist, - M_PI / 3, 0))
			: static_cast<float>(translate(d2, min_dist, max_dist, 5 * M_PI / 3, 2 * M_PI));
	}
	if (nearest_point.b == 255) {
		cout << "[INFO] Nearest point is blue." << endl;
		theta = second_nearest_point.g == 255
			? - static_cast<float>(translate(d2, min_dist, max_dist, - M_PI, - 4 * M_PI / 3))
			: static_cast<float>(translate(d2, min_dist, max_dist, M_PI / 3, 2 * M_PI / 3));
	}
	if (nearest_point.g == 255) {
		cout << "[INFO] Nearest point is green." << endl;
		theta = second_nearest_point.r == 255
			? - static_cast<float>(translate(d2, min_dist, max_dist, - (5 * M_PI / 3), - 4 * M_PI / 3))
			: static_cast<float>(translate(d2, min_dist, max_dist, M_PI, 4 * M_PI / 3));
	}

	cout << "[INFO] Second nearest point is " << (second_nearest_point.r == 255 ? "red." : second_nearest_point.g == 255 ? "green." : "blue.") << endl;
	cout << "[INFO] Estimated angle: " << theta * 180 / M_PI << " degrees" << endl;
	return theta;
}

Eigen::Matrix4f estimateInitialGuess(const pcl::PointCloud<PointT>::Ptr &source, const pcl::PointCloud<PointT>::Ptr &target)
{
	cout << "[INFO] Estimating initial guess" << endl;
	const float theta_src = estimateRotation(source);
	const float theta_tgt = estimateRotation(target);

	cout << "Difference in degrees: " << (theta_tgt - theta_src) * 180 / M_PI << endl;

	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
	transformation_matrix(0, 0) = std::cos(theta_tgt - theta_src);
	transformation_matrix(0, 2) = std::sin(theta_tgt - theta_src);
	transformation_matrix(2, 0) = -std::sin(theta_tgt - theta_src);
	transformation_matrix(2, 2) = std::cos(theta_tgt - theta_src);

	return transformation_matrix;
}

// ----------------------------------------------------------- Main Function -----------------------------------------------------------
int main()
{
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<PointT>::Ptr>> clouds_debug;
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator <pcl::PointCloud<PointT>::Ptr>> clouds_to_combine;
	const std::string colors = "RGB";

	for (int camera_index = 1; camera_index <= 2; camera_index++)
	{
		pcl::PointCloud<PointT>::Ptr calibration_cloud(new pcl::PointCloud<PointT>);
		pcl::PointCloud<PointT>::Ptr camera_cloud(new pcl::PointCloud<PointT>);

		for (int color_index = 0; color_index < colors.size(); color_index++)
		{
			// E.g. "camera_1_R.pcd", "camera_1_G.pcd", "camera_1_B.pcd"
			std::string filename = "binary_camera_" + std::to_string(camera_index) + "_" + colors[color_index] + ".pcd";
			cout << "[INFO] Fetching " << filename << "..." << endl;
			pcl::PointCloud<PointT>::Ptr curr_cloud(new pcl::PointCloud<PointT>);
			pcl::io::loadPCDFile(filename, *curr_cloud);

			cout << "[INFO] Inverting Y coordinates..." << endl;
			for (auto& point : curr_cloud->points)
			{
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
			*camera_cloud += *curr_cloud;

			cout << "[INFO] Adding " << filename << " to PointCloud object of current camera." << endl;
			*calibration_cloud += *sphere_cloud;
		}
		cout << "[INFO] Adding pre-processed view of camera " << camera_index << " to list of clouds to align." << endl;
		clouds_to_combine.push_back(calibration_cloud);
		clouds_debug.push_back(camera_cloud);
	}

	// Estimate initial guess
	Eigen::Matrix4f transformation_matrix = estimateInitialGuess(clouds_to_combine.at(0), clouds_to_combine.at(1));

	// Apply initial guess
	cout << "[INFO] Applying initial guess..." << endl;
	const pcl::PointCloud<PointT>::Ptr aligned_cloud(new pcl::PointCloud<PointT>);
	transformPointCloud(*clouds_to_combine.at(0), *aligned_cloud, transformation_matrix);

	// Iterative Closest Point algorithm
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(75);
	icp.setInputSource(aligned_cloud);
	icp.setInputTarget(clouds_to_combine.at(1));
	icp.align(*aligned_cloud);
	std::cout << "Applied " << iterations << " ICP iteration(s) in " << time_it.toc() << " ms" << std::endl;

	if (icp.hasConverged()) {
		std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
		std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
		transformation_matrix = icp.getFinalTransformation();
		print4x4Matrix(transformation_matrix);
	} else {
		PCL_ERROR("\nICP has not converged.\n");
		return (-1);
	}

	// Visualization
	cout << "[INFO] Opening Cloud Viewer..." << endl;
	viewer = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
	viewer->createViewPort(0.0, 0.0, 0.5, 0.5, vp_1);
	viewer->createViewPort(0.0, 0.5, 0.5, 1, vp_2);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, vp_3);
	showCloudsLeft(clouds_to_combine.at(0), clouds_to_combine.at(1));
	showCloudsRight(clouds_to_combine.at(1), aligned_cloud);
	std::stringstream ss;
	ss << iterations;
	std::string iterations_cnt = "ICP iterations = " + ss.str();
	viewer->addText(iterations_cnt, 10, 60, 16, 255, 255, 255, "iterations_cnt", vp_3);

	// Register space callback and set camera position and window size
	viewer->registerKeyboardCallback(&keyboardEventOccurred);
	viewer->initCameraParameters();
	viewer->setSize(1280, 1024); 

	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
		// The user pressed "space" :
		if (next_iteration)
		{
			cout << "[INFO] The user pressed space" << endl;
			// The Iterative Closest Point algorithm
			time_it.tic();
			icp.align(*aligned_cloud);
			cout << "Applied 1 ICP iteration in " << time_it.toc() << " ms" << std::endl;

			if (icp.hasConverged())
			{
				printf("\033[11A");  // Go up 11 lines in terminal output.
				printf("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
				cout << "\nICP transformation " << ++iterations << " : cloud_icp -> cloud_in" << std::endl;
				transformation_matrix *= icp.getFinalTransformation();  // WARNING /!\ This is not accurate! For "educational" purpose only!
				print4x4Matrix(transformation_matrix);  // Print the transformation between original pose and current pose

				ss.str("");
				ss << iterations;
				iterations_cnt = "ICP iterations = " + ss.str();
				viewer->updateText(iterations_cnt, 10, 60, 16, 255, 255, 255, "iterations_cnt");
				viewer->updatePointCloud(aligned_cloud, "source");
			}
			else
			{
				PCL_ERROR("\nICP has not converged.\n");
				return (-1);
			}
		}
		next_iteration = false;
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
