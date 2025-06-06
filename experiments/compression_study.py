'''
TCFR full-method EM & F1 correlation with compression ratio analysis
1. cell perspective compression correlation
2. token perspective compression correlation
'''

import pandas as pd
from TCRF_main import make_response

def calculate_cell_compression_ratio(raw_table: pd.DataFrame, filtered_df: pd.DataFrame) -> float:
    raw_cells = raw_table.shape[0] * raw_table.shape[1]
    filtered_cells = filtered_df.shape[0] * filtered_df.shape[1]
    if raw_cells == 0:
        return 0.0
    return 1.0 - (filtered_cells / raw_cells)

def calculate_token_compression_ratio(raw_table: pd.DataFrame, filtered_df: pd.DataFrame) -> float:
    raw_tokens = sum([len(str(cell).split()) for row in raw_table.values for cell in row])
    filtered_tokens = sum([len(str(cell).split()) for row in filtered_df.values for cell in row])
    if raw_tokens == 0:
        return 0.0
    return 1.0 - (filtered_tokens / raw_tokens)


def main():
    index = str(input("질문을 원하는 질문 Index를 입력하세요. : \n"))
    raw_table, filtered_df = make_response(index)
    cell_ratio = calculate_cell_compression_ratio(raw_table, filtered_df)
    token_ratio = calculate_token_compression_ratio(raw_table, filtered_df)
    print(f"Cell Compression Ratio: {cell_ratio:.2%}")
    print(f"Token Compression Ratio: {token_ratio:.2%}")

if __name__ == "__main__":
    main()