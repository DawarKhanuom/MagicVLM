
# Magic Leap 2 VLMs4XR ___ Mixed Reality + BLIP-2  with ONNX 

## Overview
This project extends Magic Leap 2’s **Camera Mixed Reality C API** sample to integrate **ONNX Runtime** and run a **BLIP-2 Vision-Language Model** (encoder + decoder) for on-device image captioning.

It combines:
- Mixed Reality camera streaming (MLSDK v1.12)
- ONNX Runtime inference on Magic Leap 2
- BLIP-2 for captions from captured frames
- Eye tracking & controller inputs for interaction

## Features
- Capture images via Magic Leap’s camera API  
- Run encoder to extract vision features  
- Run decoder to generate captions  
- Show results in the MR scene  
- CPU or GPU execution (if supported by your ORT build)  

## Project Structure



# We extneded Camera Mixed Reality Sample App

This sample demonstrates how to setup mixed reality camera and record the video from it using MediaRecorder API or capture photo.
Note that the app releases the camera only when it's destroyed, not when minimized - that allows for recording actions outside the app,
like the menu or other applications on the device.

## Prerequisites

Refer to https://developer-docs.magicleap.cloud/docs/guides/native/getting-started/native-getting-started

## Gui
 - There is a dialog that enables user to re-request permissions.
 - There is a small window that enables user to start/stop recording and view some basic info regarding current session.
 - There is a small window that enables user to capture photo and view some basic info regarding current session.

### Running on device

```sh
adb install ./app/build/outputs/apk/ml2/debug/com.magicleap.capi.sample.camera_mixed_reality-debug.apk
adb shell am start -a android.intent.action.MAIN -n com.magicleap.capi.sample.camera_mixed_reality/android.app.NativeActivity
```

### Downloading captured files from the device

To download captures:
```sh
adb pull /storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/captures
```
Please note that full filepaths for individual files will be available in logcat, also this command downloads files to the cwd.

### Removing from device

```sh
adb uninstall com.magicleap.capi.sample.camera_mixed_reality
```

## What to Expect

 - Small dialog with current session information should be visible and interactive.


## Models path 
pull /storage/emulated/0/Android/data/com.magicleap.capi.sample.camera_mixed_reality/files/models 
