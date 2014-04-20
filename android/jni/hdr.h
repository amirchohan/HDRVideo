
#ifndef JNIAPI_H
#define JNIAPI_H

#include <pthread.h>
#include <EGL/egl.h> // requires ndk r5 or newer
#include <GLES/gl.h>


extern "C" {
    JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStart(JNIEnv* jenv, jobject obj);
    JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnResume(JNIEnv* jenv, jobject obj);
    JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnPause(JNIEnv* jenv, jobject obj);
    JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStop(JNIEnv* jenv, jobject obj);
    JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeSetSurface(JNIEnv* jenv, jobject obj, jobject surface);
};

#endif // JNIAPI_H