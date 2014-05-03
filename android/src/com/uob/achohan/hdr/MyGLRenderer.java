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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGL11;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;
import javax.microedition.khronos.opengles.GL10;

import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.SurfaceHolder;

/**
 * Provides drawing instructions for a GLSurfaceView object. This class
 * must override the OpenGL ES drawing lifecycle methods:
 * <ul>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onSurfaceCreated}</li>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onDrawFrame}</li>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onSurfaceChanged}</li>
 * </ul>
 */
public class MyGLRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

	private final String TAG = "hdr";
	
	private final String vss =
		"attribute vec2 vPosition;\n" +
		"attribute vec2 vTexCoord;\n" +
		"varying vec2 texCoord;\n" +
		"void main() {\n" +
		"  texCoord = vTexCoord;\n" +
		"  gl_Position = vec4 ( vPosition.x, vPosition.y, 0.0, 1.0 );\n" +
		"}";

	private final String fss =
		"#extension GL_OES_EGL_image_external : require\n" +
		"precision mediump float;\n" +
		"uniform samplerExternalOES sTexture;\n" +
		"varying vec2 texCoord;\n" +
		"void main() {\n" +
		"  gl_FragColor = texture2D(sTexture,texCoord);\n" +
		"}";

	private int[] hTex;
	private int[] targetTex;
	private IntBuffer targetFramebuffer;
	private FloatBuffer pVertex;
	private FloatBuffer pTexCoord;
	private int hProgram;

	private Camera mCamera;
	private SurfaceTexture mSTexture;

	private boolean mUpdateST = false;

	private MyGLSurfaceView mView;

	EGL10 mEgl;
	EGLDisplay mEglDisplay;
	EGLSurface mEglSurface;
	EGLConfig mEglConfig;
	EGLContext mEglContext;

	private int image_width;
	private int image_height;


	MyGLRenderer ( MyGLSurfaceView view ) {
		mView = view;
		float[] vtmp = { 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
		float[] ttmp = { 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };
		pVertex = ByteBuffer.allocateDirect(8*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
		pVertex.put (vtmp);
		pVertex.position(0);
		pTexCoord = ByteBuffer.allocateDirect(8*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
		pTexCoord.put (ttmp);
		pTexCoord.position(0);
	}


	public void onSurfaceCreated (GL10 unused, EGLConfig config) {
		mCamera = Camera.open();

		Camera.Parameters param = mCamera.getParameters();
		param.setPictureSize(640, 480);
		mCamera.setParameters(param);

		image_width = mCamera.getParameters().getPictureSize().width;
		image_height = mCamera.getParameters().getPictureSize().height;
		Log.d(TAG, "image size: " + image_width + "x" + image_height);

		initTex();

		mSTexture = new SurfaceTexture (hTex[0]);
		mSTexture.setOnFrameAvailableListener(this);
		try {
			mCamera.setPreviewTexture(mSTexture);
		}
		catch ( IOException ioe ) {
		}

		initCL(image_width, image_height);

		GLES20.glClearColor (1.0f, 1.0f, 0.0f, 1.0f);
		hProgram = loadShader (vss, fss);
	}

	public void onDrawFrame ( GL10 unused ) {
		GLES20.glClear( GLES20.GL_COLOR_BUFFER_BIT );
		
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, targetFramebuffer.get(0));
		int fbret = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
	    if (fbret != GLES20.GL_FRAMEBUFFER_COMPLETE) {
	      Log.d(TAG, "unable to bind fbo" + fbret);
	    }
		
		GLES20.glClear( GLES20.GL_COLOR_BUFFER_BIT );

		synchronized(this) {
			if (mUpdateST) {
				mSTexture.updateTexImage();
				mUpdateST = false;
			}
		}

		GLES20.glUseProgram(hProgram);

		int ph = GLES20.glGetAttribLocation(hProgram, "vPosition");
		int tch = GLES20.glGetAttribLocation (hProgram, "vTexCoord");
		int th = GLES20.glGetUniformLocation (hProgram, "sTexture");

		GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
		GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);
		GLES20.glUniform1i(th, 0);

		GLES20.glVertexAttribPointer(ph, 2, GLES20.GL_FLOAT, false, 4*2, pVertex);
		GLES20.glVertexAttribPointer(tch, 2, GLES20.GL_FLOAT, false, 4*2, pTexCoord );
		GLES20.glEnableVertexAttribArray(ph);
		GLES20.glEnableVertexAttribArray(tch);

		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
		GLES20.glFlush();
		
		processFrame(targetTex[0]);

		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
		GLES20.glUseProgram(hProgram);

		ph = GLES20.glGetAttribLocation(hProgram, "vPosition");
		tch = GLES20.glGetAttribLocation (hProgram, "vTexCoord");
		th = GLES20.glGetUniformLocation (hProgram, "sTexture");

		GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, targetTex[0]);
		GLES20.glUniform1i(th, 0);

		GLES20.glVertexAttribPointer(ph, 2, GLES20.GL_FLOAT, false, 4*2, pVertex);
		GLES20.glVertexAttribPointer(tch, 2, GLES20.GL_FLOAT, false, 4*2, pTexCoord );
		GLES20.glEnableVertexAttribArray(ph);
		GLES20.glEnableVertexAttribArray(tch);
		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
		GLES20.glFlush();
	}

	public void onSurfaceChanged ( GL10 unused, int width, int height ) {
		GLES20.glViewport( 0, 0, width, height );
		Camera.Parameters param = mCamera.getParameters();
		List<Size> psize = param.getSupportedPreviewSizes();
		if ( psize.size() > 0 ) {
			int i;
			for ( i = 0; i < psize.size(); i++ ) {
				if ( psize.get(i).width < width || psize.get(i).height < height ) break;
			}
			if ( i > 0 ) i--;
			param.setPreviewSize(psize.get(i).width, psize.get(i).height);
		}
		param.set("orientation", "landscape");
		mCamera.setParameters(param);
		mCamera.startPreview();
	}







	private void initTex() {
		hTex = new int[1];
		targetTex = new int[1];
		GLES20.glGenTextures ( 1, hTex, 0 );
		GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

		GLES20.glGenTextures ( 1, targetTex, 0 );
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, targetTex[0]);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
		GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, image_width, image_height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
		
		targetFramebuffer = IntBuffer.allocate(1);
		GLES20.glGenFramebuffers(1, targetFramebuffer);
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, targetFramebuffer.get(0));
		GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, targetTex[0], 0);
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
	}


	public synchronized void onFrameAvailable ( SurfaceTexture st ) {
		mUpdateST = true;
		mView.requestRender();
	}

	private static int loadShader ( String vss, String fss ) {
		int vshader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
		GLES20.glShaderSource(vshader, vss);
		GLES20.glCompileShader(vshader);
		int[] compiled = new int[1];
		GLES20.glGetShaderiv(vshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
		if (compiled[0] == 0) {
			Log.e("Shader", "Could not compile vshader");
			Log.v("Shader", "Could not compile vshader:"+GLES20.glGetShaderInfoLog(vshader));
			GLES20.glDeleteShader(vshader);
			vshader = 0;
	 	}

		int fshader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
		GLES20.glShaderSource(fshader, fss);
		GLES20.glCompileShader(fshader);
		GLES20.glGetShaderiv(fshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
		if (compiled[0] == 0) {
			Log.e("Shader", "Could not compile fshader");
			Log.v("Shader", "Could not compile fshader:"+GLES20.glGetShaderInfoLog(fshader));
			GLES20.glDeleteShader(fshader);
			fshader = 0;
		}

		int program = GLES20.glCreateProgram();
		GLES20.glAttachShader(program, vshader);
		GLES20.glAttachShader(program, fshader);
		GLES20.glLinkProgram(program);
			 
		return program;
	}

	private void deleteTex() {
		GLES20.glDeleteTextures (1, hTex, 0);
	}

	public void close() {
		mUpdateST = false;
		mSTexture.release();
		mCamera.stopPreview();
		mCamera.release();
		killCL();
		deleteTex();
	}


	public void updateStatus(String text) {
		Log.d(TAG, text);
	}

	public static native void initCL(int image_width, int image_height);
	public static native void processFrame(int gl_texture_id);
	public static native void killCL();


	static {
		System.loadLibrary("hdr");
	}

}