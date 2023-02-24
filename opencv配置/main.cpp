#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/imgproc/types_c.h>
#include <iostream>  

using namespace cv;
using namespace std;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat& H, const Mat& src)
{
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

    V1 = H * V2;
    //左上角(0,0,1)
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

}

// 剔除p1中mask值为0的特征点（无效点）
void maskout_points(vector<Point2f>& p1, Mat& mask)
{
    vector<Point2f> p1_copy = p1;
    p1.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
        {
            p1.push_back(p1_copy[i]);
        }
    }
}

//剔除ransac后错误的特征点
void maskout_keypoints(vector<KeyPoint>& KeyPoints, Mat mask)
{
    vector<KeyPoint> key_copy = KeyPoints;
    KeyPoints.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
        {
            KeyPoints.push_back(key_copy[i]);
        }
    }

}

//剔除ransac后错误的匹配对
void maskout_matches(vector<DMatch>& matches, Mat mask)
{
    vector<DMatch> Match_copy = matches;
    matches.clear();
    DMatch temp;
    int count = 0;

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
        {
            temp.queryIdx = count;
            temp.trainIdx = count;
            temp.distance = Match_copy[i].distance;
            matches.push_back(temp);
            count++;
        }
    }

}


int main(int argc, char* argv[])
{

    Mat image01 = imread("data/4/test2.jpg", 1);    //右图
    Mat image02 = imread("data/4/test1.jpg", 1);    //左图


    //灰度图转换  
    Mat image1, image2;
    cvtColor(image01, image1, CV_RGB2GRAY);
    cvtColor(image02, image2, CV_RGB2GRAY);


    Ptr<Feature2D> siftdetector = cv::SIFT::create(0, 3, 0.04, 10);// SIFT提取特征点0, 3, 0.04, 10
    vector<KeyPoint> keyPoint1, keyPoint2;//特征点
    cv::Mat imageDesc1, imageDesc2;//描述子
    //特征点检测与描述，为下边的特征点匹配做准备    
    siftdetector->detectAndCompute(image1, noArray(), keyPoint1, imageDesc1);
    siftdetector->detectAndCompute(image2, noArray(), keyPoint2, imageDesc2);


    FlannBasedMatcher matcher;//创建一个特征匹配的对象
    //匹配点对容器的容器，每个容器里面装最近邻和次近邻的匹配点对
    vector<vector<DMatch> > matchePoints;
    //匹配点对容器，装匹配点对
    vector<DMatch> GoodMatchePoints;

    //向matcher传入特征描述子
    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();
    //使用k近邻(knn)查找imageDesc2的每个特征点在imageDesc1中的最近邻和次近邻点，所以最后一个参数设置为2
    matcher.knnMatch(imageDesc2, matchePoints, 2);
    cout << "total match points: " << matchePoints.size() << endl;

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchePoints.size(); i++)
    {
        //判定最近邻与次近邻的比值是不是小于alpha，是就保留该配对的最近点对
        if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
        {
            //由上可知[i][0]是最近邻，[i][1]是次近邻，满足比值的情况下保留最近邻
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }
    //可视化特征匹配及第一次剔除误匹配后的点对
    cv::Mat first_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    cv::namedWindow("first_match", WINDOW_NORMAL);//让显示窗口可以任意拖动大小
    imshow("first_match", first_match);
    imwrite("data/first_match.jpg", first_match);
    cv::waitKey(0);

    /*上述得到的特征点对是乱序的，比如图A的特征点1匹配图B的特征点54，
    * 而有的特征点又在图B中找不到对应点对
    * 为了好看和方便调试分析，只保留下那些正确匹配的特征点，并按顺序排列一一对应
    */
    vector<Point2f> imagePoints1, imagePoints2;//特征点坐标
    //剔除后的特征点容器
    vector<KeyPoint> filteredkeypoints1, filteredkeypoints2;
    vector<DMatch> goodMatchestemp;//剔除后的特征点匹配对容器
    DMatch Match;//匹配对
    int tempi = 0;
    for (int i = 0; i < GoodMatchePoints.size(); i++)  // 可能只有一个
    {
        Match.queryIdx = tempi;//从0开始按顺序一一对应
        Match.trainIdx = tempi;
        Match.distance = GoodMatchePoints[i].distance;
        tempi++;
        goodMatchestemp.push_back(Match);//匹配对装入容器
        //取出对应的特征点
        filteredkeypoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx]);
        filteredkeypoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx]);
        //取出对应的特征点纯坐标
        imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
    }
    //更新原来的匹配以及特征点
    GoodMatchePoints = goodMatchestemp;
    keyPoint1 = filteredkeypoints1;
    keyPoint2 = filteredkeypoints2;
    Mat mask;


    //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, cv::RANSAC, 3.0, mask);
    cout << "1变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵
    /*使用掩膜矩阵剔除错误的匹配点对
    本程序为了方便后面的调试分析，   将误匹配的特征点对以及相应的特征点都删除了
    留下的正确匹配对按顺序一一对应
    */
    maskout_points(imagePoints1, mask);
    maskout_points(imagePoints2, mask);

    maskout_keypoints(keyPoint1, mask);
    maskout_keypoints(keyPoint2, mask);
    maskout_matches(GoodMatchePoints, mask);
    //将RANSAC后的特征匹配可视化
    cv::Mat second_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, second_match);
    cv::namedWindow("second_match", WINDOW_NORMAL);
    imshow("second_match", second_match);
    imwrite("data/second_match.jpg", second_match);
    cv::waitKey(0);


    /*计算配准图的四个顶点坐标
    * 图像经过单应性变换后，边界可能会超出之前的范围
    * 因此只要算出四个顶点变换的像素坐标就能知道变换后图像的范围
    */
    CalcCorners(homo, image01);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    //图像配准  
    Mat imageTransform1, imageTransform2;
    warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    imshow("直接经过透视矩阵变换", imageTransform1);
    imwrite("data/trans1.jpg", imageTransform1);

    /*https://blog.csdn.net/weixin_41108706/article/details/88566069
    * Mat homo的元素是float64类型，应使用double访问
    */
    ////homo矩阵的不唯一特性，以及元素h33是尺度的特性
    //double i00 = homo.at<double>(2, 2);
    //homo.at<double>(2, 2) = 2;
    ////homo = homo * 2;
    //Mat imageTransform1_2;
    //warpPerspective(image01, imageTransform1_2, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    //imshow("homo * 2", imageTransform1_2);


    //homo.at<double>(2, 2) = 0.5;
    ////homo = homo * 2;
    //Mat imageTransform1_3;
    //warpPerspective(image01, imageTransform1_3, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    //imshow("homo * 0.5", imageTransform1_3);
    return 0;
}



