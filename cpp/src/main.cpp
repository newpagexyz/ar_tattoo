#include <iostream>
#include "glad.h"
#include "shaders/shader.h"
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

GLFWwindow* window;

const cv::Scalar lowerBound = cv::Scalar(0, 48, 80);
const cv::Scalar upperBound = cv::Scalar(20, 255, 255);

float vertices[] = {
    1.0f,  1.0f, 0.0f,    1.0f, 1.0f,
    1.0f, -1.0f, 0.0f,    1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f,    0.0f, 1.0f,
    1.0f, -1.0f, 0.0f,    1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,    0.0f, 0.0f,
    -1.0f,  1.0f, 0.0f,    0.0f, 1.0f
};

float cubeVertices[] = {
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
    0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
    0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

    -0.0f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
    0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
};

static GLuint matToTexture(const cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter) {

    GLuint textureID;
    glGenTextures(1, &textureID);


    glBindTexture(GL_TEXTURE_2D, textureID);


    if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
            magFilter == GL_LINEAR_MIPMAP_NEAREST ||
            magFilter == GL_NEAREST_MIPMAP_LINEAR ||
            magFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        std::cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << std::endl;
        magFilter = GL_LINEAR;
    }


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);



    GLenum inputColourFormat = GL_BGR;
    if (mat.channels() == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }

    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            // Internal colour format to convert to
                 mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
                 mat.rows,          // Image height i.e. 480 for Kinect in standard mode
                 0,                 // Border width in pixels (can either be 1 or 0)
                 inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,  // Image data type
                 mat.ptr());        // The actual image data itself

    if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
            minFilter == GL_LINEAR_MIPMAP_NEAREST ||
            minFilter == GL_NEAREST_MIPMAP_LINEAR ||
            minFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    return textureID;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}



void initGUI()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "opencv", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}


double distance(cv::Point a, cv::Point b)
{
    return sqrt((a.x - b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}


cv::Vec2f findOrthogonal(cv::Vec2f vector) {
    float a = vector[0]; float b = vector[1];
    float c = b*b/(a*a+b*b);
    float d = 1 - c*c;
    return cv::Vec2f(std::sqrt(c)*300, std::sqrt(d)*300);
}

//const std::vector<cv::Point3d> vector3d = {
//    cv::Point3d(0.0, 10.0, 0.0),
//    cv::Point3d(10.0, 10.0, 0.0),
//    cv::Point3d(10.0, 0.0, 0.0),
//    cv::Point3d(0.0, 0.0, 0.0)
//};


const std::vector<cv::Point3f> vector3d = {
    cv::Point3f(0.0, 10.0, 0.0),
    cv::Point3f(10.0, 10.0,0.0),
    cv::Point3f(10.0, 0.0, 0.0),
    cv::Point3f(0.0, 0.0, 0.0)
};

int main(int argc, const char * argv[])
{

    if (argc != 3) {
        std::cout << "Specify video file and pic" << std::endl;
        return 0;
    }

    std::string videofile(argv[1]); std::string picfile(argv[2]);
    

    initGUI();

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }

    Shader shader = Shader();

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    unsigned int VBO_cube, VAO_cube;
    glGenVertexArrays(1, &VAO_cube);
    glGenBuffers(1, &VBO_cube);
    glBindVertexArray(VAO_cube);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    cv::Mat frame;
    cv::VideoCapture camera(videofile);

    cv::Mat tattoo_frame;
    tattoo_frame = cv::imread(picfile);
    GLint tattoo_texture;
    tattoo_texture = matToTexture(tattoo_frame, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);

    glEnable(GL_DEPTH_TEST);

    camera >> frame;

    GLint texture;


    while (!glfwWindowShouldClose(window))
    {

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        camera >> frame;
        //        cv::waitKey();
        if (frame.empty())
        {
            std::cout << "frame empty" << "\n";
            break;
        }
        cv::flip(frame, frame, 0);

        cv::Mat hsv_frame;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);


        cv::inRange(hsv_frame, lowerBound, upperBound, hsv_frame);
        cv::Mat thresh;
        cv::threshold(hsv_frame, thresh, 170, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(thresh, contours, hierarchy,  cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);

        int max_index = 0; double max_area = 0;

        for (int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i], false);
            if (area > max_area)
            {
                max_area = area;
                max_index = i;
            }
        }

        std::vector<std::vector<cv::Point>> hull(1);
        cv::convexHull(contours[max_index], hull[0]);

        cv::Moments mu;
        mu = cv::moments(contours[max_index], false);

        int cx = mu.m10 / mu.m00;
        int cy = mu.m01 / mu.m00;

        cv::Point center_of_mass(cx, cy);

        std::vector<cv::Point> nodes(hull[0].size());
        std::vector<cv::Point> bottomTriangle(2);

        for (int i = 0; i < hull[0].size(); i++)
        {
            nodes[i] = hull[0][i];
        }

        double max_angle = 0;


        cv::Point nodeA = nodes[0];
        for (int i = nodes.size()-1; i >= 0; i--)
        {
            cv::Point nodeB = nodes[i];
            double c = distance(nodeA, nodeB);
            double a = distance(nodeB, center_of_mass);
            double b = distance(center_of_mass, nodeA);

            double gamma = std::acos(  (a*a + b*b - c*c) / (2 * a * b) );
            if ((gamma >= max_angle) && (gamma < 3.14/2))
            {
                max_angle = gamma;
                bottomTriangle[0] = cv::Point(nodeA.x, nodeA.y);
                bottomTriangle[1] = cv::Point(nodeB.x,nodeB.y);
            }
            nodeA = nodeB;
        }


        cv::Vec2f vectorA = cv::Vec2f(center_of_mass.x-bottomTriangle[0].x, center_of_mass.y-bottomTriangle[0].y);
        cv::Vec2f vectorB = cv::Vec2f(center_of_mass.x-bottomTriangle[1].x, center_of_mass.y-bottomTriangle[1].y);

        cv::Vec2f mainVector = vectorA + vectorB;
        cv::Vec2f orthogonalVector = findOrthogonal(mainVector);

        cv::Mat blank  = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
        cv::Mat blank1 = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
        cv::Mat blank2 = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);


        cv::drawContours(blank1, contours, max_index, 1);
        cv::line(blank2, center_of_mass, cv::Point((int)(center_of_mass.x + orthogonalVector[0]), (int)(center_of_mass.y+orthogonalVector[1])), 1, 1);
        cv::line(blank2, center_of_mass, cv::Point((int)(center_of_mass.x - orthogonalVector[0]), (int)(center_of_mass.y-orthogonalVector[1])), 1, 1);

        cv::bitwise_and(blank1, blank2, blank);
        std::vector<cv::Point2i> intersections;
        cv::findNonZero(blank,intersections);

                cv::circle(frame, intersections[0], 5, cv::Scalar(255, 255, 255), 5);
                cv::circle(frame, intersections[1], 5, cv::Scalar(255, 255, 255), 5);

        blank  = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
        blank2 = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);

        cv::line(blank2, center_of_mass, cv::Point((int)(center_of_mass.x - mainVector[0]/3 + orthogonalVector[0]), (int)(center_of_mass.y-mainVector[1]/3+orthogonalVector[1])), 1, 1);
        cv::line(blank2, center_of_mass, cv::Point((int)(center_of_mass.x - mainVector[0]/3 - orthogonalVector[0]), (int)(center_of_mass.y-mainVector[1]/3-orthogonalVector[1])), 1, 1);

        cv::bitwise_and(blank1, blank2, blank);
        std::vector<cv::Point2i> intersections2;
        cv::findNonZero(blank,intersections2);

                cv::circle(frame, intersections2[0], 5, cv::Scalar(255, 255, 255), 5);
                cv::circle(frame, intersections2[1], 5, cv::Scalar(255, 255, 255), 5);

        texture = matToTexture(frame, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);


        shader.use();


        glm::mat4 view          = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 projection = glm::mat4(1.0f);
        //        projection = glm::perspective(glm::radians(30.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        //        view       = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
        glm::mat4 model = glm::mat4(1.0f);
        //        // pass transformation matrices to the shader
        shader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        shader.setMat4("view", view);
        shader.setMat4("model", model);

        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        if ((intersections.size() == 2) && (intersections2.size() == 2))
        {
            std::vector<cv::Point2f> vector2d;
//            vector2d = {
//                cv::Point2d(100, 100),cv::Point2d(300, 100),cv::Point2d(300, 0),cv::Point2d(0, 0),
//            };
            vector2d.push_back(intersections[0]);
            vector2d.push_back(intersections[1]);
            vector2d.push_back(intersections2[1]);
            vector2d.push_back(intersections2[0]);



            double fx = 1.5;
            cv::Mat K = (cv::Mat_<double>(3,3) << fx*1200, 0, 0.5*1200, 0, fx*1200, 0.5*(1200-1), 0, 0, 1.0);
            std::vector<double> dist = {0,0,0,0};

            std::vector<float> rvecs; std::vector<float> tvecs;

            cv::solvePnP(vector3d, vector2d, K, dist, rvecs, tvecs);
            cv::Mat rodriguez;
            cv::Rodrigues(rvecs, rodriguez);

            glBindVertexArray(VAO_cube);
            shader.use();
            glBindTexture(GL_TEXTURE_2D, tattoo_texture);
//            cv::Mat M =
//            cv::getPerspectiveTransform(vector3d, vector2d);



//            float aaa[16] = {
//                        /    M.at<float>(0,0), M.at<float>(0,1),M.at<float>(0,2), 1.0,
//                            -1*M.at<float>(1,0), -1*M.at<float>(1,1),-1*M.at<float>(1,2), 1.0,
//                            -1*M.at<float>(2,0), -1*M.at<float>(2,1),-1*M.at<float>(2,2), 1.0,
//                            0.0, 0.0, 0.0, 1.0
//                        };
            //        model = glm::rotate(model, glm::radians(45.0f), glm::vec3(1.0f, 0.3f, 0.5f));
            float aaa[16] = {
                rodriguez.at<float>(0,0), rodriguez.at<float>(0,1),rodriguez.at<float>(0,2), tvecs[0],
                -1*rodriguez.at<float>(1,0), -1*rodriguez.at<float>(1,1),-1*rodriguez.at<float>(1,2), -1.0f*tvecs[1],
                -1*rodriguez.at<float>(2,0), -1*rodriguez.at<float>(2,1),-1*rodriguez.at<float>(2,2), -1.0f*tvecs[2],
                0.0, 0.0, 0.0, 1.0
            };

            view = glm::make_mat4(aaa);

//            view = glm::inverseTranspose(view);
//            view = glm::mat4(1.0f);
//            view = glm::translate(view, glm::vec3(0.0, 0.0, 0.5));


//            view = glm::transpose(view);

//            view = glm::mat4(1.0f);
//            view = glm::translate(view, glm::vec3(0.0f,0.0f,-0.5f));
//            view = glm::translate(view, glm::vec3(0.0, 0.0, -0.5));


//                    projection = glm::mat4(1.0f);
//                    shader.setMat4("projection", projection);
//                    shader.setMat4("model", model);
            shader.setMat4("view", view);


            glDrawArrays(GL_TRIANGLES, 0, 36);
        }



        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    glfwTerminate();
}
