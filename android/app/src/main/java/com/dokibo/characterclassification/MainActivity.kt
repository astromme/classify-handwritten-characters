package com.dokibo.characterclassification

import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.dokibo.characterclassification.databinding.ActivityMainBinding
import com.dokibo.characterclassification.ml.TrainedModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.InputStream
import java.lang.Float.min
import java.nio.ByteBuffer
import kotlin.streams.toList

fun resizedBitmapWithPadding(bitmap: Bitmap, newWidth: Int, newHeight: Int) : Bitmap {
    val scale = min(newWidth.toFloat() / bitmap.width, newHeight.toFloat() / bitmap.height)
    val scaledWidth = scale * bitmap.width
    val scaledHeight = scale * bitmap.height

    val matrix = Matrix()
    matrix.postScale(scale, scale)
    matrix.postTranslate(
        (newWidth - scaledWidth) / 2f,
        (newHeight - scaledHeight) / 2f
    )

    val outputBitmap = Bitmap.createBitmap(newWidth, newHeight, bitmap.config)
    outputBitmap.eraseColor(Color.WHITE)

    Canvas(outputBitmap).drawBitmap(
        bitmap,
        matrix,
        null
    )

    return outputBitmap
}

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var inputFeature0: TensorBuffer
    private lateinit var model: TrainedModel

    private lateinit var byteBuffer: ByteBuffer
    private lateinit var charactersIndex: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)

        binding.undoButton.setOnClickListener {
            binding.drawCharacterView.undoStroke()
        }

        binding.drawCharacterView.setOnDrawingUpdatedListener {
            bitmap ->
            runModelAndUpdateText(bitmap)
        }

        setContentView(binding.root)

        initializeModel()

        val zhongImageBitmap =
            BitmapFactory.decodeResource(applicationContext.resources, R.drawable.zhong)

        runModelAndUpdateText(zhongImageBitmap)
    }

    fun initializeModel() {
        // TODO convert this to a proper interface/class
        val ins: InputStream = resources.openRawResource(
            resources.getIdentifier(
                "characters",
                "raw", packageName
            )
        )
        charactersIndex = ins.bufferedReader(Charsets.UTF_8).lines().toList()


        byteBuffer = ByteBuffer.allocate(28 * 28 * Float.SIZE_BYTES)
        model = TrainedModel.newInstance(applicationContext)
        inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28, 1), DataType.FLOAT32)

    }

    fun runModelAndUpdateText(bitmap: Bitmap) {
        val resizedBitmap = resizedBitmapWithPadding(bitmap, 28, 28)

        for (y in 0 until resizedBitmap.height) {
            for (x in 0 until resizedBitmap.width) {
                val color = resizedBitmap.getPixel(x, y)
                val grayscale = (Color.blue(color) + Color.red(color) + Color.green(color)) / 3
                val floatGrayscaleInverted = 1 - grayscale.toFloat() / 255.0F
                Log.d("Main", "y,x: $y,$x = $grayscale -> ${floatGrayscaleInverted}")
                byteBuffer.putFloat(y * 28 + x, floatGrayscaleInverted)
            }
        }

        binding.imageView.setImageBitmap(resizedBitmap)

        // Creates inputs for reference.
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val n = 10
        // contains value and the original index pos
        // TODO pull this out into a class
        val topN = ArrayList<Pair<Float, Int>>()
        for (i in 0 until outputFeature0.flatSize) {
            if (topN.size < n) {
                topN.add(Pair(outputFeature0.floatArray[i], i))
                topN.sortBy { it.first }
            } else if (outputFeature0.floatArray[i] > topN[0].first) {
                topN.removeAt(0)
                topN.add(Pair(outputFeature0.floatArray[i], i))
                topN.sortBy { it.first }
            }
        }

        var results = ""
        topN.reversed().forEachIndexed { index, pair ->
            results += "${index+1}: ${charactersIndex[pair.second]} (${pair.first})\n"
        }
        binding.textView.text = results
    }

    override fun onDestroy() {
        super.onDestroy()

        model.close()
    }
}