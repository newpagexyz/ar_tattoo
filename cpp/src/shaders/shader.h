#ifndef SHADER_H
#define SHADER_H

#include "../glad.h"
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader
{
public:
    unsigned int ID; //Program ID
    void use();
    Shader();
    void setMat4(const std::string &name, const glm::mat4 &mat) const;
};

#endif // SHADER_H
