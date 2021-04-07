import pandas as pd

if __name__ == "__main__":
    print("Prepare 1PGA... \n Read PGA data")
    df = pd.read_excel("/home/rimichael/pro_tooling/data/pga/Nisthal_Mayo_2019_updated_3xESLyS9.xlsx")
    unique_df = df.iloc[:, [0,1]].drop_duplicates()

    fasta_str = ""
    for seq, description in zip(unique_df.iloc[:, 0], unique_df.iloc[:, 1]):
        fasta_str += f">pga_{description}\n{seq}\n"
    with open("./pga_nisthal2019_updated.fa", "w") as outfile:
        outfile.write(fasta_str)
