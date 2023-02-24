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
    double v2[] = { 0, 0, 1 };//���Ͻ�
    double v1[3];//�任�������ֵ
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

    V1 = H * V2;
    //���Ͻ�(0,0,1)
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //���½�(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //������
    V1 = Mat(3, 1, CV_64FC1, v1);  //������
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //���Ͻ�(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //������
    V1 = Mat(3, 1, CV_64FC1, v1);  //������
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //���½�(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //������
    V1 = Mat(3, 1, CV_64FC1, v1);  //������
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

}

// �޳�p1��maskֵΪ0�������㣨��Ч�㣩
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

//�޳�ransac������������
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

//�޳�ransac������ƥ���
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

    Mat image01 = imread("data/4/test2.jpg", 1);    //��ͼ
    Mat image02 = imread("data/4/test1.jpg", 1);    //��ͼ


    //�Ҷ�ͼת��  
    Mat image1, image2;
    cvtColor(image01, image1, CV_RGB2GRAY);
    cvtColor(image02, image2, CV_RGB2GRAY);


    Ptr<Feature2D> siftdetector = cv::SIFT::create(0, 3, 0.04, 10);// SIFT��ȡ������0, 3, 0.04, 10
    vector<KeyPoint> keyPoint1, keyPoint2;//������
    cv::Mat imageDesc1, imageDesc2;//������
    //����������������Ϊ�±ߵ�������ƥ����׼��    
    siftdetector->detectAndCompute(image1, noArray(), keyPoint1, imageDesc1);
    siftdetector->detectAndCompute(image2, noArray(), keyPoint2, imageDesc2);


    FlannBasedMatcher matcher;//����һ������ƥ��Ķ���
    //ƥ����������������ÿ����������װ����ںʹν��ڵ�ƥ����
    vector<vector<DMatch> > matchePoints;
    //ƥ����������װƥ����
    vector<DMatch> GoodMatchePoints;

    //��matcher��������������
    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();
    //ʹ��k����(knn)����imageDesc2��ÿ����������imageDesc1�е�����ںʹν��ڵ㣬�������һ����������Ϊ2
    matcher.knnMatch(imageDesc2, matchePoints, 2);
    cout << "total match points: " << matchePoints.size() << endl;

    // Lowe's algorithm,��ȡ����ƥ���
    for (int i = 0; i < matchePoints.size(); i++)
    {
        //�ж��������ν��ڵı�ֵ�ǲ���С��alpha���Ǿͱ�������Ե�������
        if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
        {
            //���Ͽ�֪[i][0]������ڣ�[i][1]�Ǵν��ڣ������ֵ������±��������
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }
    //���ӻ�����ƥ�估��һ���޳���ƥ���ĵ��
    cv::Mat first_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    cv::namedWindow("first_match", WINDOW_NORMAL);//����ʾ���ڿ��������϶���С
    imshow("first_match", first_match);
    imwrite("data/first_match.jpg", first_match);
    cv::waitKey(0);

    /*�����õ����������������ģ�����ͼA��������1ƥ��ͼB��������54��
    * ���е�����������ͼB���Ҳ�����Ӧ���
    * Ϊ�˺ÿ��ͷ�����Է�����ֻ��������Щ��ȷƥ��������㣬����˳������һһ��Ӧ
    */
    vector<Point2f> imagePoints1, imagePoints2;//����������
    //�޳��������������
    vector<KeyPoint> filteredkeypoints1, filteredkeypoints2;
    vector<DMatch> goodMatchestemp;//�޳����������ƥ�������
    DMatch Match;//ƥ���
    int tempi = 0;
    for (int i = 0; i < GoodMatchePoints.size(); i++)  // ����ֻ��һ��
    {
        Match.queryIdx = tempi;//��0��ʼ��˳��һһ��Ӧ
        Match.trainIdx = tempi;
        Match.distance = GoodMatchePoints[i].distance;
        tempi++;
        goodMatchestemp.push_back(Match);//ƥ���װ������
        //ȡ����Ӧ��������
        filteredkeypoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx]);
        filteredkeypoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx]);
        //ȡ����Ӧ�������㴿����
        imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
    }
    //����ԭ����ƥ���Լ�������
    GoodMatchePoints = goodMatchestemp;
    keyPoint1 = filteredkeypoints1;
    keyPoint2 = filteredkeypoints2;
    Mat mask;


    //��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, cv::RANSAC, 3.0, mask);
    cout << "1�任����Ϊ��\n" << homo << endl << endl; //���ӳ�����
    /*ʹ����Ĥ�����޳������ƥ����
    ������Ϊ�˷������ĵ��Է�����   ����ƥ�����������Լ���Ӧ�������㶼ɾ����
    ���µ���ȷƥ��԰�˳��һһ��Ӧ
    */
    maskout_points(imagePoints1, mask);
    maskout_points(imagePoints2, mask);

    maskout_keypoints(keyPoint1, mask);
    maskout_keypoints(keyPoint2, mask);
    maskout_matches(GoodMatchePoints, mask);
    //��RANSAC�������ƥ����ӻ�
    cv::Mat second_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, second_match);
    cv::namedWindow("second_match", WINDOW_NORMAL);
    imshow("second_match", second_match);
    imwrite("data/second_match.jpg", second_match);
    cv::waitKey(0);


    /*������׼ͼ���ĸ���������
    * ͼ�񾭹���Ӧ�Ա任�󣬱߽���ܻᳬ��֮ǰ�ķ�Χ
    * ���ֻҪ����ĸ�����任�������������֪���任��ͼ��ķ�Χ
    */
    CalcCorners(homo, image01);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    //ͼ����׼  
    Mat imageTransform1, imageTransform2;
    warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
    imwrite("data/trans1.jpg", imageTransform1);

    /*https://blog.csdn.net/weixin_41108706/article/details/88566069
    * Mat homo��Ԫ����float64���ͣ�Ӧʹ��double����
    */
    ////homo����Ĳ�Ψһ���ԣ��Լ�Ԫ��h33�ǳ߶ȵ�����
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



