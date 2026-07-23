# 내 기준선 실험 계획서 (1페이지)

작성일: arm64-mps 환경에서 v2_d1_06 자동 생성

## 1. 문제 정의
- **시나리오**: 사내 회의록을 기기 안에서 요약하는 비서 (문서 외부 전송 금지)
- **오프라인 필수**: 예 — 서버 fallback 불가, 온디바이스 단독 동작

## 2. 기기 제약
- **목표 기기 RAM**: 8 GB
- 모델+KV cache+앱이 이 예산 안에 들어야 함 (d1_01 메모리 예산표, d1_04 KV 스윕 참조)

## 3. 모델·정밀도 후보
- **시작 모델**: Qwen/Qwen2.5-0.5B-Instruct
- **첫 정밀도/런타임**: q4_k_m (GGUF, llama.cpp)
- 선택 근거 — 오늘(D1) 실측 요약:

```
                              model    precision          runtime  decode_tps  ttft_s  quality  file_mib  rows
                              (env)          NaN              env         NaN     NaN      NaN       NaN     1
HuggingFaceTB/SmolLM2-135M-Instruct         fp32     transformers       110.7    0.02      NaN       NaN     3
HuggingFaceTB/SmolLM2-360M-Instruct         fp32     transformers        89.9    0.02      2.0    1560.2     1
HuggingFaceTB/SmolLM2-360M-Instruct int8-torchao transformers-cpu        25.4    0.23      2.0    1560.2     1
         Qwen2.5-0.5B-Instruct-GGUF       q4_k_m        llama.cpp       271.3    0.00      2.0     468.6     1
         Qwen2.5-0.5B-Instruct-GGUF         q8_0        llama.cpp       300.4    0.00      2.0     644.4     1
              SmolLM2-135M-Instruct         fp32     transformers       112.5    0.02      0.0     621.1     1
              SmolLM2-360M-Instruct         fp32     transformers        87.3    0.02      2.0    1560.2     1
```

## 4. 성공지표와 합격선

| 지표 (공통 스키마 열) | 합격선 |
|---|---|
| `ttft_s` | 2.0초 이하 |
| `decode_tps` | 8 tok/s 이상 |
| `quality_score` | 6문항 중 4점 이상 |

- 판정 방법: 2일차 d2_03에서 위 지표를 실측 CSV로 자동 판정한다.
