# ai-rag-engine
> **목적**  
> Indexing, Retrieval, Reranker, Context 생성

데이터 인덱싱 및 하이브리드 서치를 지원하는 API입니다.  
**QE는 하지 않습니다**

---
# API
## `POST` \search
> 입력 받은 query에 대한 context를 반환합니다.
> 최대 5개의 문서를 유사도가 높은 순으로 정렬하여 반환합니다. **이 때, 각 문서의 출처를 함께 반환합니다.**
### Req
```
{
  "query": str
}
```
### Res
```
{
  "context" : list[dict]
}
```
