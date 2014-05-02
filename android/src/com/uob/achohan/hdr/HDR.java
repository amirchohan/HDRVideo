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

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.PowerManager.WakeLock;
import android.view.Window;
import android.view.WindowManager;

public class HDR extends Activity {

	private MyGLSurfaceView mView;
	private WakeLock mWL;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		// full screen & full brightness
		requestWindowFeature ( Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		mWL = ((PowerManager)getSystemService (Context.POWER_SERVICE)).newWakeLock(PowerManager.FULL_WAKE_LOCK, "WakeLock");
		mWL.acquire();
		mView = new MyGLSurfaceView(this);
		setContentView(mView);
	}
	
	@Override
	protected void onPause() {
		if ( mWL.isHeld() )	mWL.release();
		mView.onPause();
		super.onPause();
	}
			
	@Override
	protected void onResume() {
		super.onResume();
		mView.onResume();
		mWL.acquire();
	}
}