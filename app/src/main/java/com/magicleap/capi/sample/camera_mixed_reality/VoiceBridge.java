// Copyright 2025 Dawar Khan (KAUST)
// Adapted from Alpha Cephei Vosk Demo and Magic Leap C API integration

package com.magicleap.capi.sample.camera_mixed_reality;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.text.InputType;
import android.util.Log;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;
import org.vosk.LibVosk;
import org.vosk.LogLevel;
import org.vosk.Model;
import org.vosk.Recognizer;
import org.vosk.android.SpeechService;
import org.vosk.android.StorageService;

import java.io.File;
import java.util.ArrayList;
import java.util.Locale;

public class VoiceBridge {

    static { System.loadLibrary("camera_mixed_reality"); }

    // === Native callbacks ===
    private static native void nativeOnASRResult(String text);
    private static native void nativeOnASRError(String error);

    private static final String TAG = "VoiceBridge";
    private static final boolean ALWAYS_USE_VOSK = true;

    private static final String VOSK_MODEL_RELATIVE = "models/stt/model-en-us";

    private final Activity activity;

    // Android SpeechRecognizer
    private SpeechRecognizer recognizer;
    private Intent recognizerIntent;

    // Vosk
    private boolean useVosk = false;
    private boolean voskReady = false;
    private Model voskModel;
    private Recognizer voskRecognizer;
    private SpeechService voskService;

    // Text-to-Speech
    private TextToSpeech tts;
    private boolean ttsReady = false;

    public VoiceBridge(Activity activity) {
        this.activity = activity;
    }

    // ========= Lifecycle =========

    public void init() {
        activity.runOnUiThread(() -> {
            try {
                LibVosk.setLogLevel(LogLevel.INFO);
                initTTS();

                if (ALWAYS_USE_VOSK) {
                    initVosk();
                } else if (SpeechRecognizer.isRecognitionAvailable(activity)) {
                    setupAndroidSR();
                    useVosk = false;
                } else {
                    initVosk();
                }

            } catch (Throwable t) {
                Log.e(TAG, "init() failed", t);
                nativeOnASRError("init() failed: " + t);
            }
        });
    }

    public void destroy() {
        activity.runOnUiThread(() -> {
            try {
                if (recognizer != null) { recognizer.cancel(); recognizer.destroy(); recognizer = null; }
                if (voskService != null) { voskService.stop(); voskService = null; }
                if (voskRecognizer != null) { voskRecognizer.close(); voskRecognizer = null; }
                if (voskModel != null) { voskModel.close(); voskModel = null; voskReady = false; }
                if (tts != null) { tts.stop(); tts.shutdown(); tts = null; ttsReady = false; }
            } catch (Throwable t) {
                Log.e(TAG, "destroy() exception", t);
                nativeOnASRError("destroy() exception: " + t);
            }
        });
    }

    // ========= Permissions =========

    public void requestPermissions() {
        activity.runOnUiThread(() -> {
            boolean micGranted = ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
            if (!micGranted) {
                ActivityCompat.requestPermissions(activity, new String[]{ Manifest.permission.RECORD_AUDIO }, 1001);
            }
        });
    }

    // ========= STT =========

    public void startListening() {
        activity.runOnUiThread(() -> {
            try {
                if (!useVosk) {
                    if (recognizer == null) { nativeOnASRError("Android SR not ready"); return; }
                    recognizer.startListening(recognizerIntent);
                    return;
                }

                if (!voskReady || voskModel == null) { nativeOnASRError("Vosk not ready"); return; }

                if (voskRecognizer == null) voskRecognizer = new Recognizer(voskModel, 16000.0f);
                if (voskService != null) { voskService.stop(); voskService = null; }

                voskService = new SpeechService(voskRecognizer, 16000.0f);
                voskService.startListening(new org.vosk.android.RecognitionListener() {
                    @Override
                    public void onPartialResult(String hyp) {
                        String partial = extractText(hyp);
                        if (partial != null && !partial.isEmpty()) {
                            nativeOnASRResult(partial); // live partial updates
                        }
                    }

                    @Override public void onTimeout() {
                        nativeOnASRError("Timeout");
                    }

                    @Override public void onError(Exception e) {
                        nativeOnASRError("Vosk error: " + e);
                    }

                    @Override public void onResult(String hyp) {}

                    @Override public void onFinalResult(String hyp) {
                        String text = extractText(hyp);
                        nativeOnASRResult(text);
                        if (voskService != null) {
                            voskService.stop();
                            voskService = null;
                        }
                    }
                });

            } catch (Throwable t) {
                Log.e(TAG, "startListening() error", t);
                nativeOnASRError("startListening() error: " + t);
            }
        });
    }

    public void stopListening() {
        activity.runOnUiThread(() -> {
            try {
                if (!useVosk && recognizer != null) {
                    recognizer.stopListening();
                } else if (voskService != null) {
                    voskService.stop();
                    voskService = null;
                }
            } catch (Throwable t) {
                nativeOnASRError("stopListening() failed: " + t);
            }
        });
    }

    // ========= TTS =========

    public void speak(String text) {
        activity.runOnUiThread(() -> {
            try {
                if (!ttsReady || tts == null) {
                    nativeOnASRError("TTS not initialized");
                    return;
                }
                tts.speak(text == null ? "" : text, TextToSpeech.QUEUE_FLUSH, null, "magic-tts-001");
            } catch (Throwable t) {
                nativeOnASRError("speak() error: " + t);
            }
        });
    }

    private void initTTS() {
        tts = new TextToSpeech(activity.getApplicationContext(), status -> {
            ttsReady = (status == TextToSpeech.SUCCESS);
            if (ttsReady) {
                int r = tts.setLanguage(Locale.getDefault());
                if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "TTS: language not fully supported");
                }
            }
        });
    }

    // ========= SR Internals =========

    private void setupAndroidSR() {
        useVosk = false;
        recognizer = SpeechRecognizer.createSpeechRecognizer(activity);
        recognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onReadyForSpeech(Bundle b) {}
            @Override public void onBeginningOfSpeech() {}
            @Override public void onRmsChanged(float v) {}
            @Override public void onBufferReceived(byte[] b) {}
            @Override public void onEndOfSpeech() {}
            @Override public void onError(int e) { nativeOnASRError("ASR error code: " + e); }
            @Override public void onResults(Bundle res) {
                ArrayList<String> list = res.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                nativeOnASRResult((list != null && !list.isEmpty()) ? list.get(0) : "");
            }
            @Override public void onPartialResults(Bundle b) {}
            @Override public void onEvent(int e, Bundle b) {}
        });

        recognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
    }

    private void initVosk() {
        try {
            useVosk = true;
            File external = activity.getExternalFilesDir(null);
            File modelDir = new File(external, VOSK_MODEL_RELATIVE);
            if (looksLikeVoskModel(modelDir)) {
                voskModel = new Model(modelDir.getAbsolutePath());
                voskReady = true;
            } else {
                StorageService.unpack(activity, "model-en-us", "vosk", (model) -> {
                    voskModel = model;
                    voskReady = true;
                }, (ex) -> nativeOnASRError("Unpack failed: " + ex));
            }
        } catch (Throwable t) {
            nativeOnASRError("initVosk failed: " + t);
        }
    }

    private boolean looksLikeVoskModel(File dir) {
        return dir.exists() && dir.isDirectory() &&
                new File(dir, "conf/model.conf").exists() &&
                new File(dir, "am").exists();
    }

    private static String extractText(String hypothesisJson) {
        try {
            JSONObject json = new JSONObject(hypothesisJson);
            return json.optString("text", json.optString("partial", ""));
        } catch (Throwable t) {
            return "";
        }
    }

    // ========= UI helpers: virtual keyboard edit =========

    public void showKeyboard() {
        // Placeholder for parity if you later bind IME to a native view.
        // The dialog below forces focus+IME, which is usually what you want.
        activity.runOnUiThread(() -> { /* no-op */ });
    }

    public void openEditDialog(String initial) {
        activity.runOnUiThread(() -> {
            final EditText input = new EditText(activity);
            input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
            input.setText(initial == null ? "" : initial);
            input.setSelection(input.getText().length());

            AlertDialog dialog = new AlertDialog.Builder(activity)
                .setTitle("Edit question")
                .setView(input)
                .setPositiveButton("Save", (d, w) -> {
                    String text = input.getText() == null ? "" : input.getText().toString();
                    nativeOnASRResult(text);  // send back to native to update the ImGui text box
                })
                .setNegativeButton("Cancel", null)
                .create();

            dialog.show();

            // Force focus + show IME
            input.post(() -> {
                input.requestFocus();
                InputMethodManager imm = (InputMethodManager) activity.getSystemService(Activity.INPUT_METHOD_SERVICE);
                if (imm != null) imm.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT);
            });
        });
    }
}
