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

#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <android/native_window.h> // requires ndk r5 or newer
#include <EGL/egl.h> // requires ndk r5 or newer
#include <GLES/gl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>


#include "logger.h"
#include "renderer.h"

#define LOG_TAG "EglSample"

static GLint vertices[][3] = {
	{ -0x10000, -0x10000, -0x10000 },
	{  0x10000, -0x10000, -0x10000 },
	{  0x10000,  0x10000, -0x10000 },
	{ -0x10000,  0x10000, -0x10000 },
	{ -0x10000, -0x10000,  0x10000 },
	{  0x10000, -0x10000,  0x10000 },
	{  0x10000,  0x10000,  0x10000 },
	{ -0x10000,  0x10000,  0x10000 }
};

static GLint colors[][4] = {
	{ 0x00000, 0x00000, 0x00000, 0x10000 },
	{ 0x10000, 0x00000, 0x00000, 0x10000 },
	{ 0x10000, 0x10000, 0x00000, 0x10000 },
	{ 0x00000, 0x10000, 0x00000, 0x10000 },
	{ 0x00000, 0x00000, 0x10000, 0x10000 },
	{ 0x10000, 0x00000, 0x10000, 0x10000 },
	{ 0x10000, 0x10000, 0x10000, 0x10000 },
	{ 0x00000, 0x10000, 0x10000, 0x10000 }
};

static GLubyte indices[] = {
	0, 4, 5,    0, 5, 1,
	1, 5, 6,    1, 6, 2,
	2, 6, 7,    2, 7, 3,
	3, 7, 4,    3, 4, 0,
	4, 7, 6,    4, 6, 5,
	3, 0, 1,    3, 1, 2
};

using namespace hdr;

Renderer::Renderer()
	: _msg(MSG_NONE), _display(0), _surface(0), _context(0), _angle(0)
{
	status("Renderer instance created");
	pthread_mutex_init(&_mutex, 0);

	filter = new ReinhardGlobal();
	filter->setStatusCallback(updateStatus);
	return;
}

Renderer::~Renderer()
{
	status("Renderer instance destroyed");
	pthread_mutex_destroy(&_mutex);
	filter->cleanupOpenCL();
	return;
}

void Renderer::start(int texture, int width, int height)
{
	status("Creating renderer thread");
	cameraTexture = texture;

	input = {NULL, width, height};
	output = {NULL, width, height};

	cl_prop[0] = CL_GL_CONTEXT_KHR;
	cl_prop[2] = CL_EGL_DISPLAY_KHR;
	cl_prop[4] = CL_CONTEXT_PLATFORM;
	cl_prop[6] = 0;

	pthread_create(&_threadId, 0, threadStartCallback, this);
	return;
}

void Renderer::stop()
{
	status("Stopping renderer thread");

	// send message to render thread to stop rendering
	pthread_mutex_lock(&_mutex);
	_msg = MSG_RENDER_LOOP_EXIT;
	pthread_mutex_unlock(&_mutex);    

	pthread_join(_threadId, 0);
	status("Renderer thread stopped");

	return;
}

void Renderer::setWindow(ANativeWindow *window)
{
	// notify render thread that window has changed
	pthread_mutex_lock(&_mutex);
	_msg = MSG_WINDOW_SET;
	_window = window;
	pthread_mutex_unlock(&_mutex);

	return;
}

void Renderer::getImageTexture() {
	//GLuint framebuffer;
	//glGenFramebuffers(1, &framebuffer);
	//glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
//
//	////Attach 2D texture to this FBO
//	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cameraTexture, 0);
	//status("glFramebufferTexture2D() returned error %d", glGetError());
}

void Renderer::renderLoop()
{
	bool renderingEnabled = true;
	
	status("renderLoop()");

	while (renderingEnabled) {

		pthread_mutex_lock(&_mutex);

		// process incoming messages
		switch (_msg) {

			case MSG_WINDOW_SET:
				initialize();
				filter->setupOpenCL(cl_prop, params, input.width*input.height);
				params.deviceIndex = cameraTexture;		//just hardcoding it in params for now, so don't have to change the code
				filter->runOpenCL(input, output);
				break;

			case MSG_RENDER_LOOP_EXIT:
				renderingEnabled = false;
				destroy();
				break;

			default:
				break;
		}
		_msg = MSG_NONE;
		
		if (_display) {
			drawFrame();
			//filter->runOpenCL(input, output, params);
			if (!eglSwapBuffers(_display, _surface)) {
				LOG_ERROR("eglSwapBuffers() returned error %d", eglGetError());
			}
		}
		
		pthread_mutex_unlock(&_mutex);
	}
	
	status("Render loop exits");
	
	return;
}

bool Renderer::initialize()
{
	const EGLint attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_BLUE_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_RED_SIZE, 8,
		EGL_NONE
	};
	EGLDisplay display;
	EGLConfig config;    
	EGLint numConfigs;
	EGLint format;
	EGLSurface surface;
	EGLContext context;
	EGLint width;
	EGLint height;
	GLfloat ratio;
	
	status("Initializing context");
	
	if ((display = eglGetDisplay(EGL_DEFAULT_DISPLAY)) == EGL_NO_DISPLAY) {
		LOG_ERROR("eglGetDisplay() returned error %d", eglGetError());
		return false;
	}
	if (!eglInitialize(display, 0, 0)) {
		LOG_ERROR("eglInitialize() returned error %d", eglGetError());
		return false;
	}

	if (!eglChooseConfig(display, attribs, &config, 1, &numConfigs)) {
		LOG_ERROR("eglChooseConfig() returned error %d", eglGetError());
		destroy();
		return false;
	}

	if (!eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format)) {
		LOG_ERROR("eglGetConfigAttrib() returned error %d", eglGetError());
		destroy();
		return false;
	}

	ANativeWindow_setBuffersGeometry(_window, 0, 0, format);

	if (!(surface = eglCreateWindowSurface(display, config, _window, 0))) {
		LOG_ERROR("eglCreateWindowSurface() returned error %d", eglGetError());
		destroy();
		return false;
	}
	
	if (!(context = eglCreateContext(display, config, 0, 0))) {
		LOG_ERROR("eglCreateContext() returned error %d", eglGetError());
		destroy();
		return false;
	}
	
	if (!eglMakeCurrent(display, surface, surface, context)) {
		LOG_ERROR("eglMakeCurrent() returned error %d", eglGetError());
		destroy();
		return false;
	}

	if (!eglQuerySurface(display, surface, EGL_WIDTH, &width) ||
		!eglQuerySurface(display, surface, EGL_HEIGHT, &height)) {
		LOG_ERROR("eglQuerySurface() returned error %d", eglGetError());
		destroy();
		return false;
	}

	_display = display;
	_surface = surface;
	_context = context;

	cl_prop[1] = (cl_context_properties) _context;
	cl_prop[3] = (cl_context_properties) _display;

	glDisable(GL_DITHER);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	glClearColor(0, 0, 0, 0);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	
	glViewport(0, 0, width, height);

	ratio = (GLfloat) width / height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustumf(-ratio, ratio, -1, 1, 1, 10);

	return true;
}

void Renderer::destroy() {
	status("Destroying context");

	filter->cleanupOpenCL();

	eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	eglDestroyContext(_display, _context);
	eglDestroySurface(_display, _surface);
	eglTerminate(_display);
	
	_display = EGL_NO_DISPLAY;
	_surface = EGL_NO_SURFACE;
	_context = EGL_NO_CONTEXT;
	return;
}

void Renderer::drawFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0, 0, -3.0f);
	glRotatef(_angle, 0, 1, 0);
	glRotatef(_angle*0.25f, 1, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	glFrontFace(GL_CW);
	glVertexPointer(3, GL_FIXED, 0, vertices);
	glColorPointer(4, GL_FIXED, 0, colors);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_BYTE, indices);

	_angle += 1.2f;
}

void* Renderer::threadStartCallback(void *myself)
{
	Renderer *renderer = (Renderer*)myself;

	renderer->renderLoop();
	pthread_exit(0);
	
	return 0;
}



int updateStatus(const char *format, va_list args)
{
	// Generate message
	size_t sz = vsnprintf(NULL, 0, format, args) + 1;
	char *msg = (char*)malloc(sz);
	vsprintf(msg, format, args);

	__android_log_print(ANDROID_LOG_DEBUG, "hdr", "%s", msg);

	free(msg);
	return 0;
}

// Variadic argument wrapper for updateStatus
void status(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	updateStatus(fmt, args);
	va_end(args);
}