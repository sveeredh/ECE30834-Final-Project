// EXECUTION COMMAND: g++ -std=c++17 mesh_deformation.cpp imgui.cpp imgui_draw.cpp imgui_widgets.cpp imgui_tables.cpp imgui_impl_glfw.cpp imgui_impl_opengl3.cpp -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lglfw3 -lglew32 -lopengl32 -lgdi32 -limm32 -o mesh_demo.exe 
// OPEN EXE: .\mesh_demo.exe 
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cfloat>

// half edge data struct
struct HalfEdge {int id, vertex, face, next, prev, twin;};
struct Vertex {glm::vec3 position, normal; int halfedge;};
struct Face {int halfedge;};

struct HalfEdgeMesh {
    std::vector<Vertex> vertices;
    std::vector<HalfEdge> halfedges;
    std::vector<Face> faces;

    void build(const std::vector<glm::vec3>& positions, const std::vector<unsigned int>& indices) {
        vertices.assign(positions.size(), {glm::vec3(0), glm::vec3(0), -1});
        for (size_t i = 0; i < positions.size(); ++i) vertices[i].position = positions[i];
        std::unordered_map<long long, int> edgeMap;
        auto edgeKey = [](int a, int b) -> long long { return (long long)a * 1000000LL + b; };
        size_t triCount = indices.size() / 3;
        halfedges.resize(triCount * 3);
        faces.resize(triCount);
        for (size_t t = 0; t < triCount; ++t) {
            int v[3] = {(int)indices[t*3+0], (int)indices[t*3+1], (int)indices[t*3+2]};
            for (int i = 0; i < 3; ++i) {
                int heIdx = t * 3 + i;
                halfedges[heIdx] = { heIdx, v[(i+1)%3], (int)t, (int)(t*3+(i+1)%3), (int)(t*3+(i+2)%3), -1 };
                vertices[v[i]].halfedge = heIdx;
                edgeMap[edgeKey(v[i], v[(i+1)%3])] = heIdx;
            }
            faces[t].halfedge = t * 3;
        }
        for (auto& he : halfedges) {
            int from = halfedges[he.prev].vertex, to = he.vertex;
            auto it = edgeMap.find(edgeKey(to, from));
            if (it != edgeMap.end()) he.twin = it->second;
        }
        recomputeNormals();
    }

    std::vector<int> oneRing(int v) const {
        std::vector<int> ring;
        if (v < 0 || v >= (int)vertices.size() || vertices[v].halfedge < 0) return ring;
        int startHe = vertices[v].halfedge, he = startHe;
        do { ring.push_back(halfedges[he].vertex); int twin = halfedges[he].twin;
            if (twin < 0) break; he = halfedges[twin].next;
        } while (he != startHe);
        return ring;
    }

    void smoothJoint(int jointVertex, int iterations) {
        std::vector<int> area = oneRing(jointVertex);
        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<glm::vec3> nextPos(vertices.size());
            for (int i : area) {
                auto neighbors = oneRing(i);
                if (neighbors.empty()) { nextPos[i] = vertices[i].position; continue; }
                glm::vec3 centroid(0.0f);
                for (int n : neighbors) centroid += vertices[n].position;
                nextPos[i] = glm::mix(vertices[i].position, centroid / (float)neighbors.size(), 0.5f);
            }
            for (int i : area) vertices[i].position = nextPos[i];
        }
    }

    void recomputeNormals() {
        for (auto& v : vertices) v.normal = glm::vec3(0.0f);
        for (auto& f : faces) {
            int he0 = f.halfedge, he1 = halfedges[he0].next, he2 = halfedges[he1].next;
            glm::vec3 p0 = vertices[halfedges[he0].vertex].position, p1 = vertices[halfedges[he1].vertex].position, p2 = vertices[halfedges[he2].vertex].position;
            glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
            vertices[halfedges[he0].vertex].normal += n; vertices[halfedges[he1].vertex].normal += n; vertices[halfedges[he2].vertex].normal += n;
        }
        for (auto& v : vertices) if (glm::length(v.normal) > 1e-6f) v.normal = glm::normalize(v.normal);
    }

    std::vector<float> packVBO() const {
        std::vector<float> data; data.reserve(vertices.size() * 6);
        for (auto& v : vertices) {
            data.push_back(v.position.x); data.push_back(v.position.y); data.push_back(v.position.z);
            data.push_back(v.normal.x);   data.push_back(v.normal.y);   data.push_back(v.normal.z);
        }
        return data;
    }
};

// rigid joint bending
void jointBend(HalfEdgeMesh& heMesh, const std::vector<glm::vec3>& orig, int jointVertex, float angle, float threshold, bool positiveX, glm::vec3 rotAxis) {
    glm::vec3 pivot = orig[jointVertex];
    for (int i = 0; i < (int)heMesh.vertices.size(); i++) {
        glm::vec3 o = orig[i];
        bool pastJoint = o.y < (pivot.y + 0.1f); 
        bool sameSide  = positiveX ? (o.x > threshold) : (o.x < threshold);
        bool sameZ     = std::abs(o.z - pivot.z) < 3.5f; 
        if (pastJoint && sameSide && sameZ) {
            glm::vec3 offset = o - pivot;
            glm::mat4 rot = glm::rotate(glm::mat4(1.0f), angle, rotAxis);
            heMesh.vertices[i].position = pivot + glm::vec3(rot * glm::vec4(offset, 1.0f));
        }
    }
    heMesh.smoothJoint(jointVertex, 5);
}


// utils
bool loadOBJ(const std::string& path, std::vector<glm::vec3>& outP, std::vector<unsigned int>& outI) {
    std::ifstream file(path); if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok == "v") { glm::vec3 v; ss >> v.x >> v.y >> v.z; outP.push_back(v); }
        else if (tok == "f") {
            std::string vs; std::vector<int> fv;
            while (ss >> vs) fv.push_back(std::stoi(vs.substr(0, vs.find('/'))) - 1);
            for (size_t i = 1; i + 1 < fv.size(); ++i) { outI.push_back(fv[0]); outI.push_back(fv[i]); outI.push_back(fv[i+1]); }
        }
    }
    return true;
}

const char* VERT_SRC = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos; layout(location = 1) in vec3 aNormal;
out vec3 fragPos; out vec3 fragNormal;
uniform mat4 model, view, projection;
void main() {
    fragPos = vec3(model * vec4(aPos, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(fragPos, 1.0);
}
)GLSL";

const char* FRAG_SRC = R"GLSL(
#version 330 core
in vec3 fragPos; in vec3 fragNormal; out vec4 fragColor;
uniform vec3 lightPos, lightColor, objectColor;
void main() {
    vec3 N = normalize(fragNormal); vec3 L = normalize(lightPos - fragPos);
    float diff = max(dot(N, L), 0.15);
    fragColor = vec4(objectColor * lightColor * (diff + 0.1), 1.0);
}
)GLSL";

GLuint compileShader(GLuint type, const char* src) {
    GLuint s = glCreateShader(type); glShaderSource(s, 1, &src, NULL); glCompileShader(s); return s;
}

struct GPUMesh {
    GLuint vao=0, vbo=0, ibo=0; int count=0;
    void upload(const std::vector<float>& v, const std::vector<unsigned int>& i) {
        count = i.size(); glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glGenBuffers(1, &ibo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, v.size()*4, v.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.size()*4, i.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, 0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12); glEnableVertexAttribArray(1);
    }
    void update(const std::vector<float>& v) { glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferSubData(GL_ARRAY_BUFFER, 0, v.size()*4, v.data()); }
    void draw(int mode) { 
        glBindVertexArray(vao); 
        if (mode == 0) { glPointSize(2.0f); glDrawArrays(GL_POINTS, 0, count); } 
        else { glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, 0); } 
    }
};

// main
int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Mesh Deformation Tool", NULL, NULL);
    glfwMakeContextCurrent(window); glewInit(); glEnable(GL_DEPTH_TEST);

    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true); ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    std::vector<glm::vec3> positions; std::vector<unsigned int> indices;
    loadOBJ("human.obj", positions, indices);
    HalfEdgeMesh heMesh; heMesh.build(positions, indices);
    std::vector<glm::vec3> originalPositions = positions;

    float minY = FLT_MAX, maxY = -FLT_MAX;
    for (auto& v : heMesh.vertices) { minY = std::min(minY, v.position.y); maxY = std::max(maxY, v.position.y); }
    float height = maxY - minY;
    float elbowY = minY + height * 0.72f, kneeY = minY + height * 0.25f;
    int elbowR=0, elbowL=0, kneeR=0, kneeL=0;
    float bER=FLT_MAX, bEL=FLT_MAX, bKR=FLT_MAX, bKL=FLT_MAX;
    for (int i = 0; i < (int)heMesh.vertices.size(); i++) {
        glm::vec3 p = heMesh.vertices[i].position;
        if (p.x >  4.0f && std::abs(p.y - elbowY) < bER) { bER = std::abs(p.y-elbowY); elbowR = i; }
        if (p.x < -4.0f && std::abs(p.y - elbowY) < bEL) { bEL = std::abs(p.y-elbowY); elbowL = i; }
        if (p.x >  0.3f && std::abs(p.y - kneeY)  < bKR) { bKR = std::abs(p.y-kneeY);  kneeR  = i; }
        if (p.x < -0.3f && std::abs(p.y - kneeY)  < bKL) { bKL = std::abs(p.y-kneeY);  kneeL  = i; }
    }

    GPUMesh gpuMesh; gpuMesh.upload(heMesh.packVBO(), indices);
    GLuint vs = compileShader(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint prog = glCreateProgram(); glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog);

    // state vars
    float elbowAngle = 0.0f, kneeAngle = 0.0f;
    float bodyRotation = 0.0f; // manual rot
    int meshMode = 2;

    auto updateModel = [&]() {
        for (int i = 0; i < (int)heMesh.vertices.size(); i++) heMesh.vertices[i].position = originalPositions[i];
        jointBend(heMesh, originalPositions, elbowR, elbowAngle,  3.1f, true,  glm::vec3(-1,0,0));
        jointBend(heMesh, originalPositions, elbowL, elbowAngle, -3.1f, false, glm::vec3(-1,0,0));
        jointBend(heMesh, originalPositions, kneeR,  kneeAngle,   0.3f, true,  glm::vec3(1,0,0));
        jointBend(heMesh, originalPositions, kneeL,  kneeAngle,  -0.3f, false, glm::vec3(1,0,0));
        heMesh.recomputeNormals();
        gpuMesh.update(heMesh.packVBO());
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        float aspect = (float)display_w / (display_h > 0 ? (float)display_h : 1.0f);

        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        
        ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_Always); 
        ImGui::Begin("Deformation Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize);
        
        // body rot
        ImGui::SliderFloat("Body Rotation", &bodyRotation, 0.0f, 360.0f, "%.1f deg");
        
        ImGui::Separator();
        if (ImGui::SliderAngle("Elbows", &elbowAngle, 0, 150)) updateModel();
        if (ImGui::SliderAngle("Knees", &kneeAngle, 0, 150)) updateModel();
        if (ImGui::Button("Reset Pose")) { elbowAngle = 0; kneeAngle = 0; bodyRotation = 0; updateModel(); }
        
        ImGui::Separator();
        ImGui::SliderInt("Mesh Display", &meshMode, 0, 2, "Points -> Solid");
        ImGui::End();

        glClearColor(0.08f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);

        glm::mat4 m = glm::rotate(glm::mat4(1.0f), glm::radians(bodyRotation), glm::vec3(0,1,0));
        
        glm::mat4 v = glm::lookAt(glm::vec3(0,10,45), glm::vec3(0,10,0), glm::vec3(0,1,0));
        glm::mat4 p = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);

        glUniformMatrix4fv(glGetUniformLocation(prog, "model"), 1, 0, glm::value_ptr(m));
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, 0, glm::value_ptr(v));
        glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, 0, glm::value_ptr(p));
        glUniform3f(glGetUniformLocation(prog, "lightPos"), 10, 20, 40);
        glUniform3f(glGetUniformLocation(prog, "lightColor"), 1, 1, 1);
        glUniform3f(glGetUniformLocation(prog, "objectColor"), 0.3f, 0.6f, 0.7f);

        if (meshMode == 1) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        gpuMesh.draw(meshMode);

        ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    return 0;
}