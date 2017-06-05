/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.dokibo.classifier.view;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by amitshekhar on 16/03/17.
 */

public class DrawModel {

    private static final String TAG = "DrawModel";

    public static class LineElem {
        public float x;
        public float y;

        private LineElem(float x, float y) {
            this.x = x;
            this.y = y;
        }
    }

    public static class Line {
        private List<LineElem> elems = new ArrayList<>();

        private Line() {
        }

        private void addElem(LineElem elem) {
            elems.add(elem);
        }

        public int getElemSize() {
            return elems.size();
        }

        public LineElem getElem(int index) {
            return elems.get(index);
        }
    }

    public static class Extremes {
        public float minX = Float.MAX_VALUE;
        public float minY = Float.MAX_VALUE;
        public float maxX = Float.MIN_VALUE;
        public float maxY = Float.MIN_VALUE;

        public void update(float x, float y) {
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
    }

    private Line mCurrentLine;

    private int mWidth;  // pixel width = 28
    private int mHeight; // pixel height = 28

    private Extremes mExtremes = new Extremes();

    private List<Line> mLines = new ArrayList<>();

    public DrawModel(int width, int height) {
        this.mWidth = width;
        this.mHeight = height;
    }

    public int getWidth() {
        return mWidth;
    }

    public int getHeight() {
        return mHeight;
    }

    public void startLine(float x, float y) {
        mCurrentLine = new Line();
        mCurrentLine.addElem(new LineElem(x, y));
        mLines.add(mCurrentLine);
        mExtremes.update(x, y);
    }

    public void endLine() {
        mCurrentLine = null;
    }

    public void addLineElem(float x, float y) {
        if (mCurrentLine != null) {
            mCurrentLine.addElem(new LineElem(x, y));
            mExtremes.update(x, y);
        }
    }

    public int getLineSize() {
        return mLines.size();
    }

    public Line getLine(int index) {
        return mLines.get(index);
    }

    public Extremes getExtremes() { return mExtremes; }

    public void clear() {
        mLines.clear();
    }

    public void undo() {
        if (mLines.size() > 0) {
            mLines.remove(mLines.size() - 1);
            Log.d(TAG, "removed line");
        }
    }
}
