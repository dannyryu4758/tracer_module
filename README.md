> README Info
>
> > writer/email: 유호건 / hgryu@euclidsoft.co.kr
> > last_modified: 2021.07.09
> > module_version: 0.0.1

> Table of contents

- [Tracer](#nori-custom-tokenizer)
  - [Guide](#guide)
    - [Install guide](#install-guide)
    - [Usage guide](#usage-guide)
      - [References Data List](#references-data-list)
      - [Basic Function List](#function-list)
      - [Adapter Function List](#adapter-function-list)
  - [Customizing](#customizing)

# Tracer

수집데이터로 부터 분석용데이터를 만들어 내는 과정 중 전처리를 위한 다양한 기능들을 제공하는 모듈

## Guide

### Install guide

```
$ pip install (파일경로)/tracer-0.0.2.tar.gz
혹은
$ pip install git+http://euclidai.kr:20080/data_analysis/tracer_module.git
```

### Usage guide

```python
from tracer import Tracer
tracer = Tracer()
result = tracer.함수명(파라미터)
```

#### References Data List

##### special_univer_nm_dict : 특수대학(사범대, 의대, 법대) 동의어 사전

```python
# Default 특수대학(사범대, 의대, 법대) 동의어 사전 조회
tracer.get_special_univer_nm_dict()
# 특수대학(사범대, 의대, 법대) 동의어 사전 변경
tracer = Tracer(special_univer_nm_dict = 사용자 정의 사전)
```

##### etc_list : 불용어 리스트

```python
# Default 불용어 리스트 조회
tracer.get_etc_list()
# 불용어 리스트 변경
tracer = Tracer(etc_list = 사용자 정의 리스트)
```

##### need_space_list : 법인 수식어 리스트

```python
# Default 법인 수식어 리스트 조회
tracer.get_need_space_list()
# 법인 수식어 리스트 변경
tracer = Tracer(need_space_list = 사용자 정의 리스트)
```

##### word_unification_dict : 기관명 영문 약어 사전

```python
# Default 기관명 영문 약어 사전 조회
tracer.get_word_unification_dict()
# 기관명 영문 약어 사전 변경
tracer = Tracer(word_unification_dict = 사용자 정의 사전)
```

##### stopword_list : 행 데이터 삭제 조건(용어) 리스트

```python
# Default 행 데이터 삭제 조건(용어) 리스트 조회
tracer.get_stopword_list()
# 행 데이터 삭제 조건(용어) 리스트 변경
tracer = Tracer(stopword_list = 사용자 정의 리스트)
```

#### Function List

##### bracket_cleaning(param, only_bracket)

```
괄호 내 문자열 삭제 (괄호포함)
param : string
only_bracket : 괄호만 삭제 (기본값 False)
return : string / None(문자열 부재시)
```

##### not_word_del(param)

```
문자열(숫자,문자)이 없는 데이터 None 처리
param : string
return string / None(문자열 부재시)
```

##### replace_not_mean_word(param)

```
연속되는 특수문자와 공백으로 처리
ex) '# &@' =>  ' '
param : string
return : string / None(문자열 부재시)
```

##### double_space_cleaning(param)

```
두 칸이상 공백 한칸 공백으로 변경
param : string
return : string / None(문자열 부재시)
```

##### cleaning_all_agency_nm(param)

```
기관명,대학명 모두 정제
ex) '한국 기술 개발원 | 포항공과 대학교 산업 협력관'
     → '한국기술개발원 | 포항공과대학교 산업협력관'
기관명 정제 후 대학명 정제 진행
대학, 대학교, 대학원 모두 포함
기관명 데이터 개수 : 129,109개
대학, 대학교, 대학원 데이터 개수 : 1,380개
param : string
return : string / None(문자열 부재시)
```

##### cleaning_organ_nm (param)

```
기관명 정제
param : string
return : string / None(문자열 부재시)
```

##### cleaning_univer_nm (param)

```
대학명 정제
param : string
return : string / None(문자열 부재시)
```

##### etc_cleaning(param, etc)

```
기타 수식문구 정제
param : string
etc : dict ('del_word' : True or False, 'space_word' : True or False)
    - del_word : etc_list(제거 단어)내 포함 문자열 삭제 여부 (기본값 : True)
    - space_word : need_space_list(법인 수식어)내 포함 문자열 앞뒤 띄여쓰기 여부 (기본값 : True)
return : string / None(문자열 부재시)
```

##### word_unification_cleaning(param)

```
기관명 약어 정제
ex) 한국과학기술원 → KAIST
param : string
return : string / None(문자열 부재시)
```

##### del_blank_spac_fb(param, etc)

```
특수문자 앞/뒤 띄어쓰기 제거
특수문자 앞/뒤 띄어쓰기 제거
param : string
etc : dict ('f_del_space' : True or False, 'b_del_space' : True, or False, 'special_char' : '특수문자')
    - f_del_space : 특수문자 앞공백 제거 여부(기본값 True)
    - b_del_space : 특수문자 뒤공백 제거 여부(기본값 True)
    - special_char : 지정 특수문자 (기본값 모든 특수문자)
return : string / None(문자열 부재시)
```

##### stopword_data_del(df, target_cols)

```
지정 용어 포함시 데이터(행) 삭제
df : DataFrame
target_cols : 지정용어 포함여부 확인할 컬럼명 (list)
return : DataFrame
```

##### stopword_data_del2(df, target_cols)

```
지정 용어 포함시 데이터(행) 삭제
df : DataFrame
target_cols : 지정용어 포함여부 확인할 컬럼명 (list)
return : DataFrame
```

##### \__space_word_check_( param, df_list, full_name, agency_type)

```
공백 포함한 정규식으로 해당 기관/학교명 정규화
예시) “한국 기술 연구원”이라는 데이터가 있을 시
      기관코드에 있는 한국기술연구원”과 동일하게 공백 제거하여 반환함
param : string
df_list : 변환할 문자열(string / list)
full_name : df_list => full_name 추가 변환할 문자열(기본값 None)
agency_type : 데이터타입 (0 : 기관 / 1 : 학교)
return : string  / None(문자열 부재시)
```

##### \__custom_regex_replace_(param, wish_word, agency_type)

```
기관/대학명 정규화하는 정규식 처리 함수
param : string
wish_word : 변경될 단어(string)
agency_type : 0 (기관), 1 (학교)
return : string
```

##### custom_groupby(df, target_cols, params)

```
기존 DataFrame 컬럼을 모두 유지한 채로 원하는 컬럼별 그룹화 및 집계
df : DataFrame
params : (dict) {'group_cols' : list, 'agg_col': str, 'agg_type' : str}
    - group_cols : (list)그룹화할 컬럼명
    - agg_col : 집계할 컬럼명
    - agg_type : 집계명 string ("min" or "max") 기본값 "max"
return : DataFrame
```

##### custom_grouby_concat(df, target_cols, params)

```
그룹화하여 지정한 컬럼의 여러 행 데이터를 한개의 열 데이터로 변환
데이터가 없으면 concat 하지 않아 ', ,' 같은 불필요 데이터 미발생
df : DataFrame
target_cols : (list)합칠 컬럼명들
params : (dict) {'sum_cols' : list, 'sep': str, 'group_col' : list}
    - sum_cols : (list)합칠 컬럼명들
    - sep : 구분자 (기본값 : ',') 생략가능
    - group_col : (list)그룹화할 컬럼명
return : DataFrame
```

##### custom_col_concat(df, target_cols, params)

```
그룹화하여 지정한 컬럼의 여러 열을 concat 변환
데이터가 없으면 concat 하지 않아 ', ,' 같은 불필요 데이터 미발생
df : DataFrame
params : (dict) {'sum_cols' : list, 'sep': str, 'col_name' : str}
    - sum_cols : (list)합칠 컬럼명들
    - sep : 구분자 (기본값 : ',') 생략가능
    - col_name : concat 결과를 담을 컬럼명
return : DataFrame
```

##### custom_all_concat(df, target_cols, params)

```
그룹화하여 그룹화컬럼을 제와한 모든 컬럼과
열 데이터를 하나의 컬럼으로 list(dict()) 형태로 생성
df : DataFrame
params : (dict) {'group_cols' : list, 'col_name' : str}
    - group_cols : 그룹화할 컬럼명
    - col_name : concat 결과를 담을 컬럼명

return : DataFrame
```

##### start_end_with_special_char(param)

```
문자열 양 끝 특수문자 제거
param : string
return : string / None
```

#### remove_all_special_word(param) :

```
모든 특수문자 삭제 후 공백처리
param : string
return : string / None
```

#### comm_cd_join(df, target_cols, param) :

```
공통코드와 JOIN
df : DataFrame
params : (dict) {'code_map_table' : str, 'on':str, 'right_on':str, 'code_map_col' : str, 'new_col_name': str}
    - code_map_table : 사용할 공통코드명
    - on : 조인할 Main_df 컬럼명(좌우 공통시 on 만 입력)
    - right_on : 조인할 Sub_df 컬럼명 (생략가능)
    - code_map_col : 사용할 공통코드 컬럼명
    - new_col_name : 공통코드컬럼 변경명(생략시 기존 컬럼명 사용)

return : DataFrame
```

### Adapter Function List

```

migrate_composer 모듈에 사용하는 함수

```

#### drop_na(df, target_cols, params) :

```

결측값 삭제 처리
df : DataFrame
target_cols : 처리할 컬럼명
params : (dict) {'axis': int}
    - axis : 0 => 행삭제 or 1 => 열삭제 (기본값 0)
return : DataFrame

```

#### drop_duplicates(df, target_cols, params) :

```

결측값 삭제 처리
df : DataFrame
target_cols : 처리할 컬럼명
params : (dict) {'sortby': list, 'keep': 'first' or 'last' or False}
    - sortby : (list)정렬기준 (기본값 None)
    - keep (기본값 : 'first')
        1) 'first': 첫번째 중복데이터 제외한 나머지 중복데이터 삭제
        2) 'last': 마지막 중복데이터 제외한 나머지 중복데이터 삭제
return : DataFrame

```

#### fillna(df, target_cols, params) :

```

결측값 처리
df : DataFrame
target_cols : (list)처리할 컬럼명
params : (dict) {'replace' : str} - replace : 결측값을 대체할 문자열(기본값 : '')
return : DataFrame

```

#### df_query_filter(df, target_cols, params) :

```

DataFrame 필터링
df : DataFrame
* SQLLITE 구문으로 작성
params : (dict) {'query' : str } - query : DataFrame 전용 query
return : DataFrame

```

#### df_to_dict(df, target_cols, params) :

```

Dictionary 형태의 데이터 컬럼 생성
df : DataFrame
params : (dict) {'col_name' : str}
    - col_name : dict 타입 데이터를 담을 컬럼명
return : DataFrame

```

#### sort(df, target_cols, params) :

```

DataFrame 정렬
df : DataFrame
params : (dict) {'sort_by_cols' : list,  'order_by' : str }
    - sort_by_cols : (list) 정렬 기준 컬럼명
    - order_by
        1) 'asc' : 오름차순(기본값)
        2) 'desc' : 내림차순
return : DataFrame

```

#### rename_cols(df, target_cols, params) :

```

DataFrame 정렬
df : DataFrame
params : (dict) { '기존컬럼명1' : '변경할 컬럼명1', '기존컬럼명2' : '변경할 컬럼명2', ... }
return : DataFrame

```

#### adapter_etc_cleaning(df, target_cols, params) :

```

기타 수식문구 정제
df : DataFrame
target_cols : (list)정제할 컬럼명
params : (dict) { 'del_word' : bool, 'space_word' : bool} - del_word : etc_list내 포함 문자열 삭제 여부 (기본값 : True) - space_word : need_space_list내 포함 문자열 앞뒤 띄여쓰기 여부 (기본값 : True)
return : DataFrame

```

#### adapter_bracket_cleaning(df, target_cols, params) :

```

모든 괄호내 문자열(괄호포함) 삭제
df : DataFrame
target_cols : (list)정제할 컬럼명
params : (dict) {'only_bracket' : bool}
only_bracket : 괄호만 삭제 (기본값 False)
return : DataFrame

```

#### adapter_not_word_del(df, target_cols) :

```

문자열(숫자,문자)이 없는 데이터 None 처리
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_replace_not_mean_word(df, target_cols) :

```

연속되는 특수문자 공백으로 처리 ex) '#\s\*@' => '\s'
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_double_space_cleaning(df, target_cols) :

```

연속되는 특수문자 공백으로 처리 ex) '#\s\*@' => '\s'
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_cleaning_organ_nm(df, target_cols) :

```

기관명 정제
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_cleaning_univer_nm(df, target_cols) :

```

대학명 정제
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_cleaning_all_agency_nm(df, target_cols) :

```

기관명,대학명 모두 정제
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_word_unification_cleaning(df, target_cols) :

```

기관명 약어 정제 ex) 한국과학기술원 → KAIST
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_remove_all_special_word(df, target_cols) :

```

모든 특수문자 삭제 후 공백처리
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

#### adapter_start_end_with_special_char(df, target_cols) :

```

문자열 양 끝 특수문자 제거
df : DataFrame
target_cols : (list)정제할 컬럼명
return : DataFrame

```

```

```
