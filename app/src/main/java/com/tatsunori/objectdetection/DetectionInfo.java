package com.tatsunori.objectdetection;

import android.graphics.RectF;

public class DetectionInfo {
    private String mClassName;
    private float mConfidence;
    private RectF mRect;

    public DetectionInfo(String mClassName, float mConfidence, RectF mRect) {
        this.mClassName = mClassName;
        this.mConfidence = mConfidence;
        this.mRect = mRect;
    }

    public String getClassName() {
        return mClassName;
    }

    public void setClassName(String mClassName) {
        this.mClassName = mClassName;
    }

    public float getConfidence() {
        return mConfidence;
    }

    public void setConfidence(float mConfidence) {
        this.mConfidence = mConfidence;
    }

    public RectF getRect() {
        return mRect;
    }

    public void setRect(RectF mRect) {
        this.mRect = mRect;
    }
}
