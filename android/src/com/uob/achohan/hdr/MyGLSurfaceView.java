/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.uob.achohan.hdr;

import android.content.Context;
import android.graphics.Point;
import android.opengl.GLSurfaceView;
import android.view.SurfaceHolder;

/**
 * A view container where OpenGL ES graphics can be drawn on screen.
 * This view can also be used to capture touch events, such as a user
 * interacting with drawn objects.
 */
public class MyGLSurfaceView extends GLSurfaceView {

	private final MyGLRenderer mRenderer;

	MyGLSurfaceView (Context context, Point display_dim) {
		super(context);
		mRenderer = new MyGLRenderer(this, display_dim);
		setEGLContextClientVersion (2);
		setRenderer(mRenderer);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	public void surfaceCreated ( SurfaceHolder holder ) {
		super.surfaceCreated ( holder );
	}

	public void surfaceDestroyed ( SurfaceHolder holder ) {
		mRenderer.close();
		super.surfaceDestroyed ( holder );
	}

	public void surfaceChanged ( SurfaceHolder holder, int format, int w, int h ) {
		super.surfaceChanged ( holder, format, w, h );
	}

}
