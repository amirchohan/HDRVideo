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
#include "renderer.h"

#define LOG_TAG "hdr"

static ANativeWindow *window = 0;
static Renderer *renderer = 0;


JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnStart(JNIEnv* jenv, jobject obj)
{
    LOG_DEBUG("nativeOnStart");
    renderer = new Renderer();
    return;
}

JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_HDR_nativeOnResume(JNIEnv* jenv, jobject obj)
{
    LOG_INFO("nativeOnResume");
    renderer->start();
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
}


