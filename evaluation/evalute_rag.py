"""
RAG Evaluation Script

This script evaluates the performance of a Retrieval-Augmented Generation (RAG) system
using various metrics from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)
"""

import json
from typing import List, Tuple, Dict, Any

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
# 09/15/24 kimmeyh Added path where helper functions is located to the path
# Add the parent directory to the path since we work with notebooks
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helper_functions import (
    create_question_answer_from_context_chain,
    answer_question_from_context,
    retrieve_context_per_question
)

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:
    """
    Create a list of LLMTestCase objects for evaluation.

    Args:
        questions (List[str]): List of input questions.
        gt_answers (List[str]): List of ground truth answers.
        generated_answers (List[str]): List of generated answers.
        retrieved_documents (List[str]): List of retrieved documents.

    Returns:
        List[LLMTestCase]: List of LLMTestCase objects.
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4-turbo",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4-turbo",
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gpt-4-turbo",
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.
    
    Args:
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate
    
    Returns:
        Dict containing evaluation metrics
    """
    
    # Initialize LLM
    llm = ChatTongyi(model="qwen3-max")
    
    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
        请对以下检索结果进行评估。
        问题：
        【\n{question}\n】
        检索到的文档内容：
        【\n{context}\n】
        请根据以下三个标准，给出一个综合评分（1-5分，5分最高）：
        1. 相关性：这份文档与问题的相关程度有多高？
        2. 完整性：仅凭这份文档，是否包含了回答问题的必要信息？
        3. 简洁性：文档内容是否聚焦，没有大量无关信息？
        注意：你只需要对文档内容进行【一次】评分，输出一个包含三个字段的JSON对象。
        
        输出格式（只输出JSON，不要有其他内容）：
        {{"相关性": 分数, "完整性": 分数, "简洁性": 分数}}
        
        示例：
        {{"相关性": 4, "完整性": 3, "简洁性": 4}}
    """)
    
    # Create evaluation chain
    eval_chain = (
        eval_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        '''
        生成 {num_questions} 个关于气候变化的测试问题。

        严格要求：
        - 每行只输出一个问题
        - 不要有序号、不要有标题、不要有空行
        - 不要有任何解释或额外文字
        - 每个问题以问号结尾
        
        示例输出：
        地球温室效应的主要原因是什么？
        海平面上升对沿海城市有哪些影响？
        什么是碳足迹？
        
        请直接输出 {num_questions} 个问题：
        '''
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()
    
    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")

    # Evaluate each question
    results = []
    for question in questions:
        print("=" * 20)
        print("Question:")
        print(question)
        print("=" * 20)
        # Get retrieval results
        context = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in context])

        print("Context:")
        print(context_text)

        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
        print('-'*20)
        print(eval_result)
        print('-'*20)
    
    return {
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass
