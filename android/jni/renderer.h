//
// Copyright 2011 Tero Saarni
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef RENDERER_H
#define RENDERER_H

#include <pthread.h>
#include <EGL/egl.h> // requires ndk r5 or newer
#include <GLES/gl.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#include "HistEq.h"
#include "ReinhardLocal.h"
#include "ReinhardGlobal.h"
#include "GradDom.h"

using namespace hdr;


// Variadic argument wrapper for updateStatus
void status(const char *fmt, ...);
int updateStatus(const char *format, va_list args);


class Renderer {

public:
    Renderer();
    virtual ~Renderer();

    // Following methods can be called from any thread.
    // They send message to render thread which executes required actions.
    void start(int texture, int width, int height);
    void stop();
    void setWindow(ANativeWindow* window);
    
    
private:

    Filter* filter;
    Filter::Params params;

    enum RenderThreadMessage {
        MSG_NONE = 0,
        MSG_WINDOW_SET,
        MSG_RENDER_LOOP_EXIT
    };

    pthread_t _threadId;
    pthread_mutex_t _mutex;
    enum RenderThreadMessage _msg;
    
    // android window, supported by NDK r5 and newer
    ANativeWindow* _window;

    EGLDisplay _display;
    EGLSurface _surface;
    EGLContext _context;
    GLfloat _angle;

    Image input;
    Image output;
    
    int cameraTexture;

    cl_context_properties cl_prop[7];

    // RenderLoop is called in a rendering thread started in start() method
    // It creates rendering context and renders scene until stop() is called
    void renderLoop();
    void getImageTexture();

    bool initialize();
    void destroy();

    void drawFrame();

    // Helper method for starting the thread 
    static void* threadStartCallback(void *myself);

};

#endif // RENDERER_H
