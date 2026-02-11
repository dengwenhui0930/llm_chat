from app.services.knowledge_service import _knowledge_store
from app.services.retriever import retrieve


def setup_function():
    _knowledge_store.clear()


def test_empty_knowledge_returns_empty():
    assert retrieve("任何查询") == []


def test_retrieve_relevant_chunks():
    _knowledge_store["doc.txt"] = [
        "北京是中国的首都，有着悠久的历史文化",
        "上海是中国最大的经济中心城市",
        "Python是一种流行的编程语言",
        "深圳是中国的科技创新之都",
    ]
    results = retrieve("北京历史文化")
    assert len(results) > 0
    # The Beijing chunk should be the top result
    assert "北京" in results[0]


def test_top_k_limit():
    _knowledge_store["doc.txt"] = [
        "第一段关于机器学习的内容",
        "第二段关于机器学习算法的内容",
        "第三段关于机器学习模型的内容",
        "第四段关于机器学习训练的内容",
        "第五段关于完全无关的天气预报",
    ]
    results = retrieve("机器学习", top_k=3)
    assert len(results) <= 3


def test_no_match_returns_empty():
    _knowledge_store["doc.txt"] = [
        "这是一段关于烹饪的文字",
        "这是一段关于园艺的文字",
    ]
    results = retrieve("quantum computing xyz")
    assert results == []


def test_retrieves_across_multiple_files():
    _knowledge_store["a.txt"] = ["FastAPI是一个现代Web框架"]
    _knowledge_store["b.txt"] = ["FastAPI支持异步编程和自动文档生成"]
    results = retrieve("FastAPI框架")
    assert len(results) >= 1
