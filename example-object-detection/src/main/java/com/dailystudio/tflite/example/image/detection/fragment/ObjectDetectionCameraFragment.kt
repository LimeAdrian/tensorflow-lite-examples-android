package com.dailystudio.tflite.example.image.detection.fragment

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.dailystudio.devbricksx.GlobalContextWrapper
import com.dailystudio.devbricksx.development.Logger
import com.dailystudio.devbricksx.utils.ImageUtils
import com.dailystudio.devbricksx.utils.MatrixUtils
import com.dailystudio.tflite.example.common.image.AbsExampleCameraFragment
import com.dailystudio.tflite.example.common.image.AbsImageAnalyzer
import com.dailystudio.tflite.example.common.image.ImageInferenceInfo
import org.tensorflow.lite.examples.detection.tflite.Classifier
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream as OutputStream1

private class ObjectDetectionAnalyzer(rotation: Int, lensFacing: Int, val requireContext: Context)
    : AbsImageAnalyzer<ImageInferenceInfo, List<Classifier.Recognition>>(rotation, lensFacing) {

    companion object {
        private const val TF_OD_API_INPUT_SIZE = 416
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "custom-yolov4-tiny-detector_best.tflite"
        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap copy.txt"

        private const val TF_OD_FRAME_WIDTH = 640
        private const val TF_OD_FRAME_HEIGHT = 480

        private const val MAINTAIN_ASPECT = false
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.1f

        private const val PRE_SCALED_IMAGE_FILE = "pre-scaled.png"
        private const val CROPPED_IMAGE_FILE = "cropped.png"
    }

    private var classifier: Classifier? = null

    private var preScaleTransform: Matrix? = null
    private var preScaleRevertTransform: Matrix? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null

    private var croppedBitmap: Bitmap
    init {
        val cropSize = TF_OD_API_INPUT_SIZE

        croppedBitmap = Bitmap.createBitmap(
            cropSize, cropSize, Bitmap.Config.ARGB_8888
        )
    }


    override fun analyzeFrame(inferenceBitmap: Bitmap, info: ImageInferenceInfo): List<Classifier.Recognition>? {
        var results: List<Classifier.Recognition>?

        // if (hasSaved == 10) {
        //     try {
        //
        //          saveToGallery(requireContext, inferenceBitmap, "object")
        //     } catch (e: Exception) {
        //         e.printStackTrace()
        //     }
        // }
        // hasSaved++

        if (classifier == null) {
            val context = GlobalContextWrapper.context
            context?.let {
                classifier = YoloV4Classifier.create(
                    context.assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED
                )
            }

            Logger.debug("classifier created: $classifier")
        }

        var mappedResults: List<Classifier.Recognition>? = null
        classifier?.let { classifier ->
            val start = System.currentTimeMillis()
            results = classifier.recognizeImage(inferenceBitmap)
            val end = System.currentTimeMillis()

            info.inferenceTime = (end - start)

            Logger.debug("raw results: ${results.toString().replace("%", "%%")}")
            results?.let {
                mappedResults = mapRecognitions(it)
            }

        }

        return mappedResults
    }

    private fun mapRecognitions(results: List<Classifier.Recognition>): List<Classifier.Recognition> {
        val mappedRecognitions: MutableList<Classifier.Recognition> =
            mutableListOf()

        for (result in results) {
            val location = result.location
            if (location != null && result.confidence >= MINIMUM_CONFIDENCE_TF_OD_API) {
                cropToFrameTransform?.mapRect(location)
                preScaleRevertTransform?.mapRect(location)

                result.location = location
                mappedRecognitions.add(result)
            }
        }

        return mappedRecognitions
    }

    override fun createInferenceInfo(): ImageInferenceInfo {
        return ImageInferenceInfo()
    }

    override fun preProcessImage(
        frameBitmap: Bitmap?,
        info: ImageInferenceInfo
    ): Bitmap? {
        val scaledBitmap = preScaleImage(frameBitmap)

        scaledBitmap?.let {
            val cropSize = TF_OD_API_INPUT_SIZE

            val matrix = MatrixUtils.getTransformationMatrix(
                it.width,
                it.height,
                cropSize,
                cropSize,
                info.imageRotation,
                MAINTAIN_ASPECT
            )

            frameToCropTransform = matrix
            cropToFrameTransform = Matrix()
            matrix.invert(cropToFrameTransform)

            val canvas = Canvas(croppedBitmap)
            canvas.drawBitmap(it, matrix, null)

            dumpIntermediateBitmap(croppedBitmap, CROPPED_IMAGE_FILE)
        }

        // if (hasSaved2 == 10) {
        //     try {
        //
        //         saveToGallery(requireContext, croppedBitmap, "object")
        //     } catch (e: Exception) {
        //         e.printStackTrace()
        //     }
        // }
        // hasSaved2++

        return croppedBitmap
    }

    fun saveToGallery(context: Context, bitmap: Bitmap, albumName: String) {
        val filename = "${System.currentTimeMillis()}.png"
        val write: (OutputStream1) -> Boolean = {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_DCIM}/$albumName")
            }

            context.contentResolver.let {
                it.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)?.let { uri ->
                    it.openOutputStream(uri)?.let(write)
                }
            }
        } else {
            val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).toString() + File.separator + albumName
            val file = File(imagesDir)
            if (!file.exists()) {
                file.mkdir()
            }
            val image = File(imagesDir, filename)
            write(FileOutputStream(image))
        }
    }

    var hasSaved = 0
    var hasSaved2 = 0

    private fun preScaleImage(frameBitmap: Bitmap?): Bitmap? {
        if (frameBitmap == null) {
            return null
        }

        val matrix = MatrixUtils.getTransformationMatrix(
            frameBitmap.width, frameBitmap.height,
            TF_OD_FRAME_WIDTH, TF_OD_FRAME_HEIGHT, 0, true
        )

        preScaleTransform = matrix
        preScaleRevertTransform = Matrix()
        matrix.invert(preScaleRevertTransform)

       val scaledBitmap = ImageUtils.createTransformedBitmap(frameBitmap, matrix)

        dumpIntermediateBitmap(scaledBitmap, PRE_SCALED_IMAGE_FILE)

        return scaledBitmap
    }

}



class ObjectDetectionCameraFragment : AbsExampleCameraFragment<ImageInferenceInfo, List<Classifier.Recognition>>() {

    override fun createAnalyzer(
        screenAspectRatio: Int,
        rotation: Int,
        lensFacing: Int,
        requireContext: Context
    )
            : AbsImageAnalyzer<ImageInferenceInfo, List<Classifier.Recognition>> {
        return ObjectDetectionAnalyzer(rotation, lensFacing, requireContext)
    }

}