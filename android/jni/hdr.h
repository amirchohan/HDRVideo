
#ifndef JNIAPI_H
#define JNIAPI_H

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

extern "C" {
	// Variadic argument wrapper for updateStatus
	void status(const char *fmt, ...);
	int updateStatus(const char *format, va_list args);

	Filter* filter;
	Filter::Params params;

    cl_context_properties cl_prop[7];


	//JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStart(JNIEnv* jenv, jobject obj);
	//JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnResume(JNIEnv* jenv, jobject obj, jint cameraTexture, jint width, jint height);
	//JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnPause(JNIEnv* jenv, jobject obj);
	//JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStop(JNIEnv* jenv, jobject obj);
	//JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeSetSurface(JNIEnv* jenv, jobject obj, jobject surface);

	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_initCL(JNIEnv* jenv, jobject obj, jint width, jint height, jint in_tex, jint out_tex);
	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_processFrame(JNIEnv* jenv, jobject obj, jint input_texid, jint output_texid);
	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_killCL(JNIEnv* jenv, jobject obj);

};

#endif // JNIAPI_H