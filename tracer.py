# from base.D_dbUtills import DbConnector
import pandas as pd
import re
import logging
import traceback
from multiprocessing import cpu_count, Pool, Manager, Process
import numpy as np
import logging
from pandas.io import json
from references_file.tracer_list import etc_list_r, need_space_list_r, word_unification_dict_r, stopword_list_r, stopword_list_r2, special_univer_nm_dict_r
import os
from pandasql import sqldf


from sqlalchemy.sql.expression import column
base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.ERROR)

m = Manager()


def common_func(df, target_col, fun, multi_list, params):
    if params and len(params):
        df[target_col] = df[target_col].apply(fun, params)
    else:
        df[target_col] = df[target_col].apply(fun)
    multi_list.append(df)


def lambda_parallelize_dataframe(df, func, target_col, fun, params=None):
    multi_list = m.list()
    num_cores = int(cpu_count()/3)
    print(f"cpu_count : {num_cores}")
    df_split = np.array_split(df, num_cores)
    procs = []

    for index, one_df in enumerate(df_split):
        proc = Process(target=func, args=(
            one_df, target_col, fun, multi_list, params))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    index = 0
    result_df = pd.DataFrame()
    for df in multi_list:
        if index == 0:
            result_df = multi_list[0]
        else:
            result_df = pd.concat([result_df, df], ignore_index=True)
        index += 1
    return result_df


class Tracer():
    def __init__(self, etc_list=None, need_space_list=None, word_unification_dict=None, stopword_list=None, stopword_list2=None, special_univer_nm_dict=None):
        """
        etc_list : 불용어 리스트
        need_space_list : 법인 수식어 리스트
        word_unification_dict : 기관명 영어 약어 사전
        stopword_list : 행 데이터 삭제 조건(용어) 리스트
        special_univer_nm_dict : 특수대학(사범대,의대,법대) 동의어 사전
        """
        if not etc_list:
            self.etc_list = etc_list_r.copy()
        else:
            self.etc_list = etc_list
        if not need_space_list:
            self.need_space_list = need_space_list_r.copy()
        else:
            self.need_space_list = need_space_list
        if not word_unification_dict:
            self.word_unification_dict = word_unification_dict_r.copy()
        else:
            self.word_unification_dict = word_unification_dict
        if not stopword_list:
            self.stopword_list = stopword_list_r.copy()
        else:
            self.stopword_list = stopword_list
        if not stopword_list2:
            self.stopword_list2 = stopword_list_r2.copy()
        else:
            self.stopword_list2 = stopword_list2
        if not special_univer_nm_dict:
            self.special_univer_nm_dict = special_univer_nm_dict_r.copy()
        else:
            self.special_univer_nm_dict = special_univer_nm_dict

        self.agency_info_df = self.read_parquet(
            "agency_info.parquet").fillna("")
        self.all_univer_df = self.read_parquet(
            "all_univer_info.parquet").fillna("")

    def read_parquet(self, file_name):
        tmp_df = pd.read_parquet(f"{base_path}/parquet_file/{file_name}")
        return tmp_df

    def get_etc_list(self):
        """
        불용어 리스트
        type : list
        """
        return self.etc_list

    def get_need_space_list(self):
        """
        기관 법인 수식어 리스트
        type : list
        """
        return self.need_space_list

    def get_word_unification_dict(self):
        """
        기관 영어 약어 사전
        type : dict
        """
        return self.word_unification_dict

    def get_stopword_list(self):
        """
        포함시 데이터(행) 삭제 용어 리스트
        type : list
        """
        return self.stopword_list

    def get_special_univer_nm_dict(self):
        """
        특수대학(사범대,의대,법대) 사전
        type : dict
        """
        return self.special_univer_nm_dict
    # 데이터 정제 ==========================================================================================

    # 괄호내 문자열(괄호포함) 삭제
    def bracket_cleaning(self, param, only_bracket=False):
        """
        모든 괄호내 문자열(괄호포함) 삭제
        param : string
        only_bracket : 괄호만 삭제 (기본값 False)
        return : string / None(문자열 부재시)
        """
        try:
            if isinstance(param, str):
                if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):

                    bracket_cleaning_regx = re.compile(
                        r'\(.*\)|\[.*\]|\{.*\}|\<.*\>|\〔.*\〕|\〈.*\〉|\《.*\》|\「.*\」|\『.*\』|\【.*\】')
                    if only_bracket:
                        bracket_cleaning_regx = re.compile(
                            r'[\(\)\[\]\{\}\<\>\〔\〕\〈\〉\《\》\「\」\『\』\【\】]')
                    if bracket_cleaning_regx.search(str(param)):
                        param = bracket_cleaning_regx.sub(' ', param)
                        param = self.double_space_cleaning(param)
                else:
                    param = None
            else:
                if param and str(param).replace(" ", "") and re.search(r"[\d\w가-힣]", str(param)):

                    bracket_cleaning_regx = re.compile(
                        r'\(.*\)|\[.*\]|\{.*\}|\<.*\>|\〔.*\〕|\〈.*\〉|\《.*\》|\「.*\」|\『.*\』|\【.*\】')
                    if only_bracket:
                        bracket_cleaning_regx = re.compile(
                            r'[\(\)\[\]\{\}\<\>\〔\〕\〈\〉\《\》\「\」\『\』\【\】]')
                    if bracket_cleaning_regx.search(str(param)):
                        param = bracket_cleaning_regx.sub(' ', str(param))
                        param = self.double_space_cleaning(str(param))
                else:
                    param = None
            return param

        except:
            logging.error(traceback.format_exc())

    def not_word_del(self, param):
        """
        문자열(숫자,문자)이 없는 데이터 None 처리
        param : string
        return string / None(문자열 부재시)
        """
        try:
            if not isinstance(param, str):
                if not (param and str(param).replace(" ", "") and re.search(r"[\d\w가-힣]", str(param))):
                    param = None
            else:
                if not (param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param)):
                    param = None
            return param

        except:
            logging.error(traceback.format_exc())

    def replace_not_mean_word(self, param):
        """
        연속되는 특수문자 공백으로 처리
        ex) '#\s*@' => '\s'
        param : string
        return : string / None(문자열 부재시)
        """
        try:
            if isinstance(param, str):
                if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                    not_mean_word_regx = re.compile(
                        r'([^\s\d\w가-힣]+\s*[^\s\d\w가-힣]+)')
                    if not_mean_word_regx.search(param):
                        param = not_mean_word_regx.sub(' ', param)
                        param = self.double_space_cleaning(param)
                else:
                    param = None
            else:
                if param and str(param).replace(" ", "") and re.search(r"[\d\w가-힣]", str(param)):
                    not_mean_word_regx = re.compile(
                        r'([^\s\d\w가-힣]+\s*[^\s\d\w가-힣]+)')
                    if not_mean_word_regx.search(str(param)):
                        param = not_mean_word_regx.sub(' ', str(param))
                        param = self.double_space_cleaning(str(param))
                else:
                    param = None
            return param
        except:
            logging.error(traceback.format_exc())

    # 두칸이상 공백 한칸 공백으로 변경

    def double_space_cleaning(self, param):
        """
        두칸이상 공백 한칸 공백으로 변경
        param : string
        return : string / None(문자열 부재시)
        """
        try:
            if isinstance(param, str):
                if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                    space_regx = re.compile(r'\s{2,}')
                    param = space_regx.sub(' ', param)
                    param.strip()
                else:
                    param = None
            else:
                if param and str(param).replace(" ", "") and re.search(r"[\d\w가-힣]", str(param)):
                    space_regx = re.compile(r'\s{2,}')
                    param = space_regx.sub(' ', str(param))
                    param.strip()
                else:
                    param = None

            return param
        except:
            logging.error(traceback.format_exc())

    def cleaning_all_agency_nm(self, param):
        """
        기관명,대학명 모두 정제
        param : string
        return : string / None
        """
        # 기관명 먼저 정규화
        param = self.cleaning_organ_nm(param)
        param = self.cleaning_univer_nm(param)
        return param

    def cleaning_organ_nm(self, param):
        """
        기관명 정제
        param : string
        return : string / None(문자열 부재시)
        """
        try:
            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                for agency_name in self.agency_info_df["agency_name"]:
                    if agency_name in param.replace(" ", ""):
                        param = self._space_word_check_(param, agency_name)
            else:
                param = None
            return param
        except:
            logging.error(traceback.format_exc())

    def cleaning_univer_nm(self, param):
        """
        대학명 정제
        param : string
        return : string / None(문자열 부재시)
        """
        try:
            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                basic_regx = re.compile(r'(대(?:학교|학원|학)?)')
                special_regex_dic = {}
                for dic in self.special_univer_nm_dict:
                    temp = '|'.join(self.special_univer_nm_dict[dic])
                    regex_str = r'((?:' + temp + r')' + r'(?:\s*학교|학)?)'
                    special_regex_dic[dic] = regex_str
                if basic_regx.search(param.replace(" ", "")):
                    for full_name, abbr_name in zip(self.all_univer_df["agency_name"], self.all_univer_df["agency_addr_nm"]):
                        # 사범대, 의대, 법대명 정규화
                        for regex_dict in special_regex_dic.items():
                            special_regex = re.compile(regex_dict[1])
                            if special_regex.search(full_name) and special_regex.search(param.replace(" ", "")) and abbr_name in special_regex.sub(regex_dict[0], param.replace(" ", "")):
                                param = special_regex.sub(regex_dict[0], param)
                                break
                        # 대학명 정규화
                        regx_str = abbr_name + r"(?:학교|학)?"
                        new_basic_regx = re.compile(regx_str)
                        if abbr_name.endswith("대") and new_basic_regx.search(param.replace(" ", "")):
                            param = self._space_word_check_(
                                param, abbr_name, full_name, 1)
                        elif full_name.endswith("대학원") and full_name in param.replace(" ", ""):
                            param = self._space_word_check_(param, full_name)
            else:
                param = None
            return param
        except:
            logging.error(traceback.format_exc())

    # 기타 수식문구 정제
    def etc_cleaning(self, param, etc):
        """
        기타 수식문구 정제
        param : string
        etc : (dict) { 'del_word' : bool,  'space_word' : bool}
            - del_word : etc_list내 포함 문자열 삭제 여부 (기본값 : True)
            - space_word : need_space_list내 포함 문자열 앞뒤 띄여쓰기 여부 (기본값 : True)

        return : string / None(문자열 부재시)

        """
        del_word = True
        space_word = True
        if "del_word" in etc.keys() and etc["del_word"]:
            del_word = etc["del_word"]
        if "space_word" in etc.keys() and etc["space_word"]:
            space_word = etc["space_word"]
        try:
            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                # 제거 문구
                if del_word:
                    for etc_word in self.etc_list:
                        if param and etc_word in param:
                            param = param.replace(etc_word, "")

                # 앞뒤 띄어쓰기 문구
            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                if space_word:
                    param = self._space_word_check_(
                        param, self.need_space_list)
                    for word in self.need_space_list:
                        if word in param:
                            param = param.replace(word, f' {word} ').strip()
                            param = self.double_space_cleaning(param)
            else:
                param = None
            param = self.double_space_cleaning(param)
            return param
        except:
            logging.error(traceback.format_exc())

    # 기관명 약어 통일
    def word_unification_cleaning(self, param):
        """
        기관명 약어 정제
        ex) 한국과학기술원 → KAIST
        param : string
        return : string / None(문자열 부재시)
        """
        try:
            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                for n in self.word_unification_dict.keys():
                    if n in param.replace(' ', ''):
                        tmp_regx_str = r"\s*"
                        for m in list(n):
                            tmp_regx_str += m + r"\s*"
                        tmp_regx = re.compile(tmp_regx_str)
                        param = tmp_regx.sub(f' {n} ', param).strip()
                param = self.double_space_cleaning(param)
            else:
                param = None
            return param
        except:
            logging.error(traceback.format_exc())

    def del_blank_spac_fb(self, param, etc):
        """
        특수문자 앞/뒤 띄어쓰기 제거
        param : string
        front_del_space : 특수문자 앞공백 제거 여부(기본값 True)
        back_del_space : 특수문자 뒤공백 제거 여부(기본값 True)
        special_char : 지정 특수문자 (기본값 모든 특수문자)
        return : string / None(문자열 부재시)
        """
        f_del_space = True
        b_del_space = True
        special_char = None

        if "f_del_space" in etc.keys():
            f_del_space = etc["f_del_space"]
        if "b_del_space" in etc.keys():
            b_del_space = etc["b_del_space"]
        if "special_char" in etc.keys():
            special_char = etc["special_char"]

        try:
            char_regex = r"([^\d\w\s가-힣]+)"
            if special_char and special_char.replace(" ", ""):
                char_regex = r"(" + re.escape(special_char) + r")"

            if f_del_space:
                char_regex = r"\s*" + char_regex
            if b_del_space:
                char_regex = char_regex + r"\s*"

            if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
                param = re.sub(char_regex, r"\1", param)
                param = self.double_space_cleaning(param)
            else:
                param = None
            return param
        except:
            logging.error(traceback.format_exc())

    def remove_all_special_word(self, param):
        """
        모든 특수문자 삭제 후 공백처리
        """
        if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
            param = re.sub(r"[^a-zA-Z0-9가-힣\s]", " ", param)
        return param

    # 불필요 데이터 삭제 용어
    def stopword_data_del(self, df, target_cols):
        """
        지정 용어 포함시 데이터(행) 삭제
        df : DataFrame
        target_cols : list 지정용어 포함여부 확인할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            if col and col.replace(" ", ""):
                for stopword in self.stopword_list:
                    df = df[~df[col].str.contains(stopword, na=False)]
        return df.reset_index(drop=True)

    # 불필요 데이터 삭제 용어
    def stopword_data_del2(self, df, target_cols):
        """
        지정 용어 포함시 데이터(행) 삭제
        df : DataFrame
        column_name : 지정용어 포함여부 확인할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            if col and col.replace(" ", ""):
                for stopword in self.stopword_list2:
                    df = df[~df[col].str.contains(stopword, na=False)]
        return df.reset_index(drop=True)

    def _space_word_check_(self, param, df_list, full_name=None, agency_type=0):
        """
        공백 포함한 정규식으로 해당 기관(학교)명 정규화함
        param : 문자열(string)
        df_list : 변환할 문자열(string / list)
        full_name : df_list => full_name 추가 변환할 문자열(기본값 None)
        agency_type : 데이터타입 (0 : 기관 / 1 : 학교)
        return : string  / None(문자열 부재시)
        """
        if param and param.replace(" ", "") and re.search(r"[\d\w가-힣]", param):
            if type(df_list) is list:
                df_list.sort(key=len)
                for nsp_word in list(df_list):
                    if nsp_word.replace(" ", "") in param.replace(" ", ""):
                        param = self._custom_regex_replace_(
                            param, nsp_word, agency_type)
            elif type(df_list) is str:
                param = self._custom_regex_replace_(
                    param, df_list, agency_type)
                if full_name and full_name.replace(" ", "") and re.search(r"[\d\w가-힣]", full_name):
                    param = param.replace(df_list, full_name)
            param = self.double_space_cleaning(param)
            param = self.del_blank_spac_fb(param)
        else:
            param = None
        return param

    def _custom_regex_replace_(self, param, wish_word, agency_type):
        """
        기관/대학명 정규화하는 정규식 처리 함수
        param : string
        wish_word : 변경될 단어(string)
        agency_type : 0 (기관), 1 (학교)
        return : string
        """
        tmp_regx_str = r"\s*"
        for m in list(wish_word.replace(" ", "")):
            tmp_regx_str += re.escape(m) + r"\s*"
        if agency_type != 0:
            tmp_regx_str += r"(학\s*교\s*|학\s*)"
        tmp_regx = re.compile(tmp_regx_str)
        param = tmp_regx.sub(f' {wish_word} ', param).replace(
            wish_word, f" {wish_word} ").strip()
        return param

    def custom_groupby(self, df, target_cols=None, params=None):
        """
        기존 DataFrame 컬럼을 모두 유지한 채로 원하는 컬럼별 그룹화 및 집계
        df : DataFrame
        params : (dict) {'group_cols' : list, 'agg_col': str, 'agg_type' : str}
            - group_cols : (list)그룹화할 컬럼명
            - agg_col : 집계할 컬럼명
            - agg_type : 집계명 string ("min" or "max") 기본값 "max"

        return : DataFrame
        """
        group_col = params["group_cols"]
        agg_col = params["agg_col"]
        agg_type = "max"
        if "agg_type" in params.keys() and params["agg_type"]:
            agg_type = params["agg_type"]

        ascending_yn = False
        if agg_type == "min":
            ascending_yn = True
        index_name = df.index.name
        if not index_name:
            index_name = "index"
        temp_df = df.copy().reset_index(drop=False)
        temp_df = temp_df.groupby(group_col, as_index=False).apply(lambda x: x[x[agg_col] == x.sort_values(
            by=agg_col, ascending=ascending_yn, na_position='last')[:1][agg_col].values[0]])
        result_df = df[df.index.isin(
            temp_df[index_name])].reset_index(drop=True)
        return result_df

    def custom_groupby_concat(self, df, target_cols=None, params=None):
        """
        그룹화하여 지정한 컬럼의 여러 행 데이터를 한개의 열 데이터로 변환
        데이터가 없으면 concat 하지 않아 ', ,' 같은 불필요 데이터 미발생
        df : DataFrame
        params : (dict) {'sum_cols' : list, 'sep': str, 'group_col' : list}
            - sum_cols : (list)합칠 컬럼명들
            - sep : 구분자 (기본값 : ',') 생략가능
            - group_col : (list)그룹화할 컬럼명

        return : DataFrame
        """
        sep = ","
        group_col = params["group_col"]
        sum_cols = params["sum_cols"]
        if "sep" in params.keys() and params["sep"]:
            sep = params["sep"]
        if not sum_cols or len(sum_cols) == 0:
            sum_cols = list(df.columns)
            del sum_cols[sum_cols.index(item for item in group_col)]
        original_col_nm = {}
        for col1 in group_col:
            for index, col2 in enumerate(sum_cols):
                if col1 == col2:
                    original_col_nm[f"sub_sum_{col1}"] = col1
                    df[f"sub_sum_{col1}"] = df[col1]
                    sum_cols[index] = f"sub_sum_{col1}"
        result_df = df.copy().reset_index(drop=False)
        def custom_agg(x): return sep.join([item for item in x if item and item.replace(
            " ", "") and re.search(r"[\d\w가-힣]", item)])
        agg_dict = {}
        result_df[sum_cols] = result_df[sum_cols].fillna("").astype(str)
        for agg_col in sum_cols:
            agg_dict[agg_col] = custom_agg
        result_df = result_df.groupby(group_col).agg(agg_dict)
        result_df = result_df.reset_index(drop=False)
        result_df = pd.merge(df, result_df, how="left", on=group_col,
                             suffixes=("", "_rigtht"))
        for col in sum_cols:
            result_df[col] = result_df[f"{col}_rigtht"]
            if col.startswith("sub_sum_"):
                result_df[original_col_nm[col]] = result_df[col]
                result_df = result_df.drop([col], axis=1)
            result_df = result_df.drop([f"{col}_rigtht"], axis=1)
        return result_df

    def custom_col_concat(self, df, target_cols=None, params=None):
        """
        그룹화하여 지정한 컬럼의 여러 열을 concat 변환
        데이터가 없으면 concat 하지 않아 ', ,' 와 같은 불필요 데이터가 생기지 않음
        df : DataFrame
        params : (dict) {'sum_cols' : list, 'sep': str, 'col_name' : str}
            - sum_cols : (list)합칠 컬럼명들
            - sep : 구분자 (기본값 : ',') 생략가능
            - col_name : concat 결과를 담을 컬럼명

        return : DataFrame
        """
        sep = ","
        target_col_nm = params["col_name"]
        sum_cols = params["sum_cols"]
        if "sep" in params.keys() and params["sep"]:
            sep = params["sep"]
        result_df = df.copy()
        result_df = result_df.fillna("").astype(str)
        def custom_agg(x): return sep.join([item for item in x if item and item.replace(
            " ", "") and re.search(r"[\d\w가-힣]", item)])
        result_df[target_col_nm] = result_df[sum_cols].apply(
            custom_agg, axis=1)
        return result_df

    def custom_all_concat(self, df, target_cols=None, params=None):
        """
        그룹화하여 그룹화컬럼을 제와한 모든 컬럼과
        열 데이터를 하나의 컬럼으로 list(dict()) 형태로 생성
        df : DataFrame
        params : (dict) {'group_cols' : list, 'col_name' : str, 'remove_col_yn' : str}
            - group_cols : (list)그룹화할 컬럼명
            - col_name : concat 결과를 담을 컬럼명
            - remove_col_yn : concat 후 group_cols 삭제 여부(Y or N)
        return : DataFrame
        """
        result_df = df.copy()
        target_col_nm = params["col_name"]
        group_cols = params["group_cols"]
        rest_cols = list(set(result_df.columns.tolist())-set(group_cols))
        result_df[target_col_nm] = result_df[rest_cols].to_dict('records')
        result_df = result_df.groupby(group_cols)[target_col_nm].apply(
            list).to_frame().reset_index()
        result_df = pd.merge(df, result_df, how="left", on=group_cols,
                             suffixes=("", "_rigtht"))
        for col in list(result_df.columns):
            if col.endswith("_right"):
                result_df.drop(col, axis=1)
        if "remove_col_yn" in params.keys() and params["remove_col_yn"].upper() == "Y":
            result_df = result_df.drop(rest_cols, axis=1)
        return result_df

    def start_end_with_special_char(self, param):
        """
        문자열 양 끝 특수문자 제거
        param : string
        return : string / None
        """
        if self.not_word_del(param):
            st_ed_regex = re.compile(r"(^[^\d\wㄱ-ㅎ가-힣]+)|([^\d\wㄱ-ㅎ가-힣]+$)")
            param = st_ed_regex.sub("", param)
            if param:
                param = param.strip()
        else:
            param = None
        return param

# =========================================================================================================
    def add_groupby_rank(self, df, target_cols=None, params=None):
        """
        그룹으로 정렬 후 rank 지정
        df : DataFrame
        target_cols : (list)정렬할기준 컬럼들
        params : (dict) {'ranked_col' : str, 'group_col': str, 'new_col':str}
            - ranked_col : rank의 기준 컬럼명
            - group_col : 그룹으로 지정할 컬럼
            - new_col : rank를 넣을 새로운 컬럼명

        return : DataFrame
        """
        ranked_col = params["ranked_col"]
        group_col = params["group_col"]
        new_col = params["new_col"]
        df = df.sort_values(by=target_cols).reset_index(drop=True)
        df[new_col] = df.groupby(group_col, as_index=False)[
            ranked_col].rank(method='first')
        df[df[group_col].isnull()][new_col] = 0
        return df

    def copy_cols(self, df, target_cols=None, params=None):
        """
        특정 컬럼을 복사
        df : DataFrame
        target_cols : (list)복사되어 들어갈 컬럼명
        params : (dict) {'source_cols' : list}
            - source_cols : 복사할 컬럼명

        return : DataFrame
        """
        source_col = params["source_cols"]
        target_col = target_cols
        df[target_col] = df[source_col]
        return df

    def remove_cols(self, df, target_cols=None, params=None):
        """
        컬럼(열) 삭제
        df : DataFrame
        target_cols : (list)삭제할 컬럼

        return : DataFrame
        """
        df = df.drop(target_cols, axis=1)
        return df

    def comm_cd_join(self, df, target_cols=None, params=None):
        """
        공통코드와 JOIN
        df : DataFrame
        params : (dict) {'code_map_table' : str, 'on':str, 'right_on':str, 'code_map_col' : str, 'new_col_name': str}
            - code_map_table : 사용할 공통코드명
            - on : 조인할 Main_df 컬럼명(좌우 공통시 on 만 입력)
            - right_on : 조인할 Sub_df 컬럼명 (생략가능)
            - code_map_col : 사용할 공통코드 컬럼명
            - new_col_name : 공통코드컬럼 변경명(생략시 기존 컬럼명 사용)

        return : DataFrame
        """
        if "right_on" in params.keys() and params["right_on"]:
            df = pd.merge(df, params["code_map_table"], how="left",
                          left_on=params["on"], right_on=params["right_on"])
            df = df.drop(params["right_on"], axis=1)
        else:
            df = pd.merge(df, params["code_map_table"],
                          how="left", on=params["on"])

        if "new_col_name" in params.keys() and params["new_col_name"]:
            df = df.rename(
                columns={params["code_map_col"]: params["new_col_name"]})

        return df

    def drop_na(self, df, target_cols=None, params=None):
        """
        결측값 삭제 처리
        df : DataFrame
        target_cols : (list)처리할 컬럼명
        params : (dict) {'axis': int}
            - axis : 0 => 행삭제 or 1 => 열삭제 (기본값 0)

        return : DataFrame
        """
        axis = 0
        if "axis" in params.keys() and params["axis"]:
            axis = params["axis"]
        result_df = df.copy()
        result_df = self.adapter_not_word_del(result_df, target_cols)
        if axis == 1:
            temp = result_df[target_cols].isnull().sum()
            for col in target_cols:
                if temp[col] > 0:
                    result_df = result_df.drop([col], axis=1)
        else:
            result_df = result_df.dropna(
                axis=axis, subset=target_cols)

        return result_df

    def drop_duplicates(self, df, target_cols=None, params=None):
        """
        중복값 제거
        df : DataFrame
        target_cols : (list)중복값 처리할 컬럼명(그룹)
        params : (dict) {'sortby': list, 'keep': 'first' or 'last'}
            - sortby : (list)정렬기준 (기본값 None)
            - keep (기본값 : 'first')
                1) 'first': 첫번째 중복데이터 제외한 나머지 중복데이터 삭제
                2) 'last': 마지막 중복데이터 제외한 나머지 중복데이터 삭제

        return : DataFrame
        """
        sortby = None
        keep = "first"
        if params:
            if "sortby" in params.keys() and len(params["sortby"]) > 0:
                sortby = params["sortby"]
                df = df.sort_values(by=sortby, axis=0).reset_index(drop=True)
            if "keep" in params.keys():
                keep = params["keep"]
            df = df.drop_duplicates(
                target_cols, keep=keep, ignore_index=True)
        else:
            df = df.drop_duplicates(target_cols, ignore_index=True)
        return df

    def fillna(self, df, target_cols=None, params=None):
        """
        결측값 처리
        df : DataFrame
        target_cols : (list)처리할 컬럼명
        params : (dict) {'replace' : str}
            - replace : 결측값을 대체할 문자열(기본값 : '')

        return : DataFrame
        """

        # if params and "replace" in params.keys() and params["replace"]:
        if params and "replace" in params.keys() and not params["replace"] is None:
            replace_val = params["replace"]
            for col in target_cols:
                if replace_val == []:
                    df[col] = df[col].apply(
                        lambda x: x if isinstance(x, list) else replace_val)
                elif replace_val == {}:
                    df[col] = df[col].apply(
                        lambda x: x if isinstance(x, dict) else replace_val)
                else:
                    df[col] = df[col].fillna(replace_val)
        elif not params or not "replace" in params.keys() or params["replace"] is None:
            for col in target_cols:
                df[col] = df[col].replace(np.nan, None)
        return df

    def dict_to_json(self, param):
        param = json.dumps(param, ensure_ascii=False)
        return param
    
    def col_dict_to_json(self, df, target_cols=None, params=None):
        for col in target_cols:
            df[col] = df[col].map(self.dict_to_json)
        return df
        
    def adapter_dict_to_json(self, df, target_cols, params=None):
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.dict_to_json, params)
        return df

    def df_query_filter(self, df, target_cols=None, params=None):
        """
        DataFrame 필터링
        df : DataFrame
        params : (dict) {'query' : str }
            - query : DataFrame 전용 query

        return : DataFrame
        """
        df = df
        if "query" in params.keys():
            df = sqldf(params["query"], locals())
        return df

    def df_to_dict(self, df, target_cols=None, params=None):
        """
        행 데이터를 Dictionary 타입의 하나의 컬럼으로 생성
        df : DataFrame
        params : (dict) {'col_name' : str}
            - col_name : dict 타입 데이터를 담을 컬럼명

        return : DataFrame
        """
        new_col = params["col_name"]
        df[new_col] = df.apply(lambda x: x.to_dict(), axis=1)
        return df

    def sort(self, df, target_cols=None, params=None):
        """
        DataFrame 정렬
        df : DataFrame
        params : (dict) {'sort_by_cols' : list,  'order_by' : str }
            - sort_by_cols : (list) 정렬 기준 컬럼명
            - order_by
                1) 'asc' : 오름차순(기본값)
                2) 'desc' : 내림차순

        return : DataFrame
        """
        ascending = True
        sort_by_cols = params["sort_by_cols"]
        if params and "order_by" in params.keys() and params["order_by"]:
            if params["order_by"] == "desc":
                ascending = False
        df = df.sort_values(
            sort_by_cols, ascending=ascending).reset_index(drop=True)
        return df

    def rename_cols(self, df, target_cols=None, params=None):
        """
        컬럼명 변경
        df : DataFrame
        params : (dict) { '기존컬럼명1' : '변경할 컬럼명1', '기존컬럼명2' : '변경할 컬럼명2', ... }
        return : DataFrame
        """
        df = df.rename(columns=params)
        return df

    def adapter_etc_cleaning(self, df, target_cols, params=None):
        """
        기타 수식문구 정제
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        params : (dict) { 'del_word' : bool, 'space_word' : bool}
            - del_word : etc_list내 포함 문자열 삭제 여부 (기본값 : True)
            - space_word : need_space_list내 포함 문자열 앞뒤 띄여쓰기 여부 (기본값 : True)

        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.etc_cleaning, params)
        return df

    def adapter_bracket_cleaning(self, df, target_cols, params=None):
        """
        모든 괄호내 문자열(괄호포함) 삭제
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        params : (dict) {'only_bracket' : bool}
            - only_bracket : 괄호만 삭제 (기본값 False)

        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.bracket_cleaning, params)
        return df

    def adapter_not_word_del(self, df, target_cols):
        """
        문자열(숫자,문자)이 없는 데이터 None 처리
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.not_word_del)
        return df

    def adapter_replace_not_mean_word(self, df, target_cols):
        """
        연속되는 특수문자 공백으로 처리 ex) '#\s*@' => '\s'
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.replace_not_mean_word)
        return df

    def adapter_double_space_cleaning(self, df, target_cols):
        """
        두칸이상 공백 한칸 공백으로 변경
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.double_space_cleaning)
        return df

    def adapter_cleaning_organ_nm(self, df, target_cols):
        """
        기관명 정제
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.cleaning_organ_nm)
        return df

    def adapter_cleaning_univer_nm(self, df, target_cols):
        """
        대학명 정제
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.cleaning_univer_nm)
        return df

    def adapter_cleaning_all_agency_nm(self, df, target_cols):
        """
        기관명,대학명 모두 정제
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.cleaning_all_agency_nm)
        return df

    def adapter_word_unification_cleaning(self, df, target_cols):
        """
        기관명 약어 정제 ex) 한국과학기술원 → KAIST
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.word_unification_cleaning)
        return df

    def adapter_remove_all_special_word(self, df, target_cols):
        """
        모든 특수문자 삭제 후 공백처리
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.remove_all_special_word)
        return df

    def adapter_start_end_with_special_char(self, df, target_cols):
        """
        문자열 양 끝 특수문자 제거
        df : DataFrame
        target_cols : (list)정제할 컬럼명
        return : DataFrame
        """
        for col in target_cols:
            df = lambda_parallelize_dataframe(
                df, common_func, col, self.start_end_with_special_char)
        return df
