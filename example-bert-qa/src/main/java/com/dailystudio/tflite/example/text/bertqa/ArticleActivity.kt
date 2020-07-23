package com.dailystudio.tflite.example.text.bertqa

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.dailystudio.devbricksx.development.Logger
import com.dailystudio.tflite.example.common.AbsExampleActivity
import com.dailystudio.tflite.example.common.InferenceInfo
import com.dailystudio.tflite.example.text.bertqa.fragment.ArticleQAFragment
import com.dailystudio.tflite.example.text.bertqa.model.ArticleViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.bertqa.ml.QaClient

class ArticleActivity : AbsExampleActivity<InferenceInfo, Void>() {

    companion object {

        const val EXTRA_ARTICLE_ID = "article-id"

    }

    private var article: Article? = null
    private var qaFragment: ArticleQAFragment? = null

    override fun onResume() {
        super.onResume()

        parseArticle(intent)
    }

    private fun parseArticle(intent: Intent) {
        val articleId = intent.getIntExtra(EXTRA_ARTICLE_ID, -1)
        Logger.debug("article id: $articleId")
        if (articleId == -1) {
            return
        }

        val viewModel = ViewModelProvider(this).get(ArticleViewModel::class.java)

        article = viewModel.getArticle(articleId)
        Logger.debug("open article: $article")

        setExampleTitle(article?.getDisplayTitle() ?: title)

        qaFragment?.setArticle(article)
    }

    override fun createBaseFragment(): Fragment {
        val fragment = ArticleQAFragment()

        qaFragment = fragment

        return fragment
    }

    override fun createResultsView(): View? {
        return null
    }

    override fun createSettingsView(): View? {
        return null
    }

    override fun onResultsUpdated(results: Void) {
    }

}