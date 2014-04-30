package com.uob.achohan.hdr;

import java.io.IOException;
import java.util.List;

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Bundle;
import android.widget.Toast;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.View.OnClickListener;
import android.util.Log;


public class HDR extends Activity implements SurfaceHolder.Callback, SurfaceTexture.OnFrameAvailableListener
{

	private static String TAG = "hdr";

	private Camera mCamera;
	private SurfaceTexture mSTexture;
	private int[] cameraTexture;
	private boolean mUpdateST = false;    
	
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		Log.i(TAG, "onCreate()");
		
		setContentView(R.layout.main);
		SurfaceView surfaceView = (SurfaceView)findViewById(R.id.surfaceview);
		surfaceView.getHolder().addCallback(this);
		surfaceView.setOnClickListener(new OnClickListener() {
				public void onClick(View view) {
					Toast toast = Toast.makeText(HDR.this,
												 "This demo combines Java UI and native EGL + OpenGL renderer",
												 Toast.LENGTH_LONG);
					toast.show();
				}});
	}

	@Override
	protected void onStart() {
		super.onStart();
		Log.d(TAG, "onStart()");
		nativeOnStart();
	}

	@Override
	protected void onResume() {
		super.onResume();
		Log.i(TAG, "onResume()");
	}
	
	@Override
	protected void onPause() {
		super.onPause();
		Log.i(TAG, "onPause()");
		nativeOnPause();
	}

	@Override
	protected void onStop() {
		super.onStop();
		Log.d(TAG, "onStop()");
		nativeOnStop();
		close();
	}


	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
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
		mCamera.setParameters ( param );
		mCamera.startPreview();
		nativeSetSurface(holder.getSurface());
	}

	public void surfaceCreated(SurfaceHolder holder) {
		initTex();
		mSTexture = new SurfaceTexture(cameraTexture[0]);
		mSTexture.setOnFrameAvailableListener(this);

		mCamera = Camera.open();
		try {
			mCamera.setPreviewTexture(mSTexture);
		}
		catch ( IOException ioe ) {
		}

		GLES20.glClearColor (1.0f, 1.0f, 0.0f, 1.0f);

		Camera.Size dim = mCamera.getParameters().getPreviewSize();
		nativeOnResume(cameraTexture[0], dim.width, dim.height);
	}

	public void surfaceDestroyed(SurfaceHolder holder) {
		nativeSetSurface(null);
	}

	public void updateStatus(String text) {
		Log.d(TAG, text);
	}

	private void initTex() {
		cameraTexture = new int[1];
		cameraTexture[0] = 1;	//so that when it recreates the texture, the id doesn't change because it has already been passed to OpenCL
		GLES20.glGenTextures(1, cameraTexture, 0);
		GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture[0]);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
		Log.d(TAG, "cameraTexture initialised");
	}


	private void deleteTex() {
		GLES20.glDeleteTextures(1, cameraTexture, 0);
	}

	public void close() {
		mSTexture.release();
		//mCamera.stopPreview();
		mCamera.release();
		deleteTex();
	}

	public static native void nativeOnStart();
	public static native void nativeOnResume(int cameraTexture, int width, int height);
	public static native void nativeOnPause();
	public static native void nativeOnStop();
	public static native void nativeSetSurface(Surface surface);

	static {
		System.loadLibrary("hdr");
	}

	@Override
	public void onFrameAvailable(SurfaceTexture surfaceTexture) {
		// TODO Auto-generated method stub
		
	}
}