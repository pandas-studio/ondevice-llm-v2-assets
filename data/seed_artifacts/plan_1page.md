# 내 기준선 실험 계획서 (1페이지)

작성일: T4 환경에서 v2_d1_06 자동 생성

## 1. 문제 정의
- **시나리오**: 매장 키오스크에서 주문 문의에 답하는 안내 챗봇 (네트워크 불안정 대비)
- **오프라인 필수**: 아니오 — 서버 fallback 병행 가능(라우팅 설계 필요)

## 2. 기기 제약
- **목표 기기 RAM**: 4 GB
- 모델+KV cache+앱이 이 예산 안에 들어야 함 (d1_01 메모리 예산표, d1_04 KV 스윕 참조)

## 3. 모델·정밀도 후보
- **시작 모델**: HuggingFaceTB/SmolLM2-360M-Instruct
- **첫 정밀도/런타임**: int8 (torchao, CPU)
- 선택 근거 — 오늘(D1) 실측 요약:

```
                              model     precision          runtime  decode_tps  ttft_s  quality  file_mib  rows
                              (env)           NaN              env         NaN     NaN      NaN       NaN     1
HuggingFaceTB/SmolLM2-135M-Instruct          fp32     transformers       110.7    0.02      NaN       NaN     3
HyperCLOVAX-SEED-Text-Instruct-0.5B          fp16     transformers        24.8    0.04      4.0    1296.1     1
         Qwen/Qwen2.5-0.5B-Instruct          fp16     transformers         8.3    0.16      4.0    1201.9     1
         Qwen/Qwen2.5-0.5B-Instruct      int8-bnb     transformers         6.0    0.17      5.0     601.0     1
         Qwen/Qwen2.5-0.5B-Instruct  int8-torchao transformers-cpu        20.6   33.93      2.0    2403.9     1
         Qwen/Qwen2.5-0.5B-Instruct       nf4-bnb     transformers        13.6    0.10      2.0     430.4     1
              Qwen2.5-0.5B-Instruct          fp16     transformers        29.4    0.04      4.0    1201.9     1
         Qwen2.5-0.5B-Instruct-GGUF        q4_k_m        llama.cpp       271.3    0.00      2.0     468.6     1
         Qwen2.5-0.5B-Instruct-GGUF          q8_0        llama.cpp       300.4    0.00      2.0     644.4     1
                         Qwen3-0.6B          fp16     transformers        20.4    0.06      0.0    1433.6     1
                         Qwen3-0.6B  fp16-nothink     transformers        20.4    0.06      NaN       NaN     1
                         Qwen3-0.6B fp16-thinking     transformers        19.0    0.06      NaN       NaN     1
              SmolLM2-360M-Instruct          fp16     transformers        23.4    0.05      3.0     780.1     1
```

## 4. 성공지표와 합격선

| 지표 (공통 스키마 열) | 합격선 |
|---|---|
| `peak_mem_mib` | 1500 MiB 이하 |
| `ttft_s` | 1.0초 이하 |
| `quality_score` | 6문항 중 3점 이상 |

- 판정 방법: 2일차 d2_03에서 위 지표를 실측 CSV로 자동 판정한다.
