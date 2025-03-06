import os
from typing import Dict, Any

import datasets
from evaluate import Metric, EvaluationModuleInfo, MetricInfo

def get_metrics() -> Metric:
    return E2E_NLG_Metrics()

class E2E_NLG_Metrics(Metric):
    def _download_and_prepare(self, dl_manager):
        os.system('pip install -r matplotlib scikit-image future')
        os.system('curl -L https://cpanmin.us | perl - App::cpanminus')
        os.system('cpanm XML::Twig')

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        inputs = kwargs.get('inputs', [''] * len(references))
        python = kwargs.get('python', True)

        data = {
            'ref':references,
            'sys':predictions,
            'src':inputs if kwargs.get('allow_inputs', False) else None,
        }
        from src.metrics.e2e_nlg.measure_scores import run_coco_eval, run_pymteval, run_mteval
        coco_eval = run_coco_eval(data['ref'], data['sys'])
        scores = {metric: score for metric, score in list(coco_eval.eval.items())}

        if python:
            mteval_scores = run_pymteval(data['ref'], data['sys'])
        else:
            mteval_scores = run_mteval(data['ref'], data['sys'], data['src'])
        scores.update(mteval_scores)

        return scores

    def _info(self) -> EvaluationModuleInfo:
        return MetricInfo(
            description="a metric for e2e_nlg",
            citation="https://github.com/tuetschek/e2e-metrics",
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="pred"),
                        "references": datasets.Value("string", id="ref"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="pred"),
                        "references": datasets.Value("string", id="ref"),
                        "inputs": datasets.Value("string", id="input"),
                    }
                ),
            ]
        )


