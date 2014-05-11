// improsa.cpp (ImProSA)
// Copyright (c) 2014, James Price and Simon McIntosh-Smith,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include <stdint.h>
#include <jni.h>
#include <android/bitmap.h>
#include <android/native_window.h> // requires ndk r5 or newer
#include <android/native_window_jni.h> // requires ndk r5 or newer

#include "hdr.h"
#include "logger.h"
//#include "renderer.h"

#define LOG_TAG "hdr"


JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_initCL(JNIEnv* jenv, jobject obj, jint width, jint height, jint in_tex, jint out_tex) {
	filter = new ReinhardGlobal();
	filter->setStatusCallback(updateStatus);

	EGLDisplay mEglDisplay;
	EGLContext mEglContext;

	if ((mEglDisplay = eglGetCurrentDisplay()) == EGL_NO_DISPLAY) {
		status("eglGetCurrentDisplay() returned error %d", eglGetError());
	}

	if ((mEglContext = eglGetCurrentContext()) == EGL_NO_CONTEXT) {
		status("eglGetCurrentContext() returned error %d", eglGetError());
	}

	cl_prop[0] = CL_GL_CONTEXT_KHR;
	cl_prop[1] = (cl_context_properties) mEglContext;
	cl_prop[2] = CL_EGL_DISPLAY_KHR;
	cl_prop[3] = (cl_context_properties) mEglDisplay;
	cl_prop[4] = CL_CONTEXT_PLATFORM;
	cl_prop[6] = 0;

	params.opengl = true;

	filter->setImageSize(width, height);
	filter->setImageTextures(in_tex, out_tex);
	filter->setupOpenCL(cl_prop, params);

	return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_processFrame(JNIEnv* jenv, jobject obj, jint input_texid, jint output_texid) {
	filter->runOpenCL(input_texid, output_texid);
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_killCL(JNIEnv* jenv, jobject obj) {
	filter->cleanupOpenCL();
}




int updateStatus(const char *format, va_list args) {
	// Generate message
	size_t sz = vsnprintf(NULL, 0, format, args) + 1;
	char *msg = (char*)malloc(sz);
	vsprintf(msg, format, args);

	__android_log_print(ANDROID_LOG_DEBUG, "hdr", "%s", msg);

	free(msg);
	return 0;
}

// Variadic argument wrapper for updateStatus
void status(const char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	updateStatus(fmt, args);
	va_end(args);
}


/*static ANativeWindow *window = 0;
static Renderer *renderer = 0;


JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStart(JNIEnv* jenv, jobject obj)
{
	LOG_DEBUG("nativeOnStart");
	renderer = new Renderer();
	return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnResume(JNIEnv* jenv, jobject obj, jint cameraTexture, jint width, jint height)
{
	LOG_INFO("nativeOnResume");
	renderer->start(cameraTexture, width, height);
	return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnPause(JNIEnv* jenv, jobject obj)
{
	LOG_INFO("nativeOnPause");
	renderer->stop();
	return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStop(JNIEnv* jenv, jobject obj)
{
	LOG_INFO("nativeOnStop");
	delete renderer;
	renderer = 0;
	return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeSetSurface(JNIEnv* jenv, jobject obj, jobject surface)
{
	if (surface != 0) {
		window = ANativeWindow_fromSurface(jenv, surface);
		LOG_INFO("Got window %p", window);
		renderer->setWindow(window);
	} else {
		LOG_INFO("Releasing window");
		ANativeWindow_release(window);
	}

	return;
}*/


