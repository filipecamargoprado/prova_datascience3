from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# Caminhos relativos à pasta onde o script está localizado
PASTA_BASE = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_BASE / "modelo_breast_cancer.joblib"
# Nomes oficiais das colunas conforme a documentação da UCI
COLUNAS_UCI = [
    "Class",
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]
ATRIBUTOS = [col for col in COLUNAS_UCI if col != "Class"]
COLUNAS_CATEGORICAS = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "breast",
    "breast-quad",
    "irradiat",
]
COLUNAS_NUMERICAS = ["deg-malig"]
# Opções válidas para orientar e validar a entrada na inferência
OPCOES_ATRIBUTOS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "menopause": ["lt40", "ge40", "premeno"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
    "node-caps": ["yes", "no"],
    "deg-malig": ["1", "2", "3"],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"],
}
# =============================================================================
# MÓDULO DE TREINAMENTO
# =============================================================================
def encontrar_arquivo_base(pasta: Path = PASTA_BASE) -> Path:
    """Procura o arquivo da base Breast Cancer na pasta do script (ou subpastas)."""
    nomes_esperados = [
        "breast-cancer.data",
        "breast-cancer.csv",
        "breast_cancer.data",
        "breast_cancer.csv",
    ]
    for nome in nomes_esperados:
        caminho = pasta / nome
        if caminho.is_file():
            return caminho
    # Caso o arquivo esteja dentro de uma subpasta extraída do zip
    for nome in nomes_esperados:
        encontrados = list(pasta.rglob(nome))
        if encontrados:
            return encontrados[0]
    for extensao in ("*.data", "*.csv"):
        encontrados = sorted(pasta.glob(extensao))
        if encontrados:
            return encontrados[0]
    raise FileNotFoundError(
        f"Base de dados não encontrada em '{pasta}'. "
        "Coloque o arquivo 'breast-cancer.data' na mesma pasta deste script."
    )
def carregar_dados(caminho_arquivo: Optional[Path] = None) -> pd.DataFrame:
    """Carrega a base tratando '?' como nulo e ajusta os nomes das colunas."""
    caminho = caminho_arquivo or encontrar_arquivo_base()
    with open(caminho, "r", encoding="utf-8") as arquivo:
        primeira_linha = arquivo.readline().strip().lower()
    tem_cabecalho = "age" in primeira_linha and "menopause" in primeira_linha
    if tem_cabecalho:
        df = pd.read_csv(caminho, na_values=["?"], skipinitialspace=True)
        mapa_colunas = {
            "class": "Class",
            "tumor_size": "tumor-size",
            "inv_nodes": "inv-nodes",
            "node_caps": "node-caps",
            "deg_malig": "deg-malig",
            "breast_quad": "breast-quad",
            "irradiate": "irradiat",
        }
        df = df.rename(columns={c: mapa_colunas.get(c.strip().lower(), c.strip()) for c in df.columns})
    else:
        df = pd.read_csv(caminho, header=None, na_values=["?"], skipinitialspace=True)
        # Na base UCI original, 'Class' vem na primeira coluna; em alguns CSVs vem na última
        if df.iloc[:, 0].astype(str).str.contains("recurrence-events", na=False).any():
            df.columns = COLUNAS_UCI
        else:
            df.columns = ATRIBUTOS + ["Class"]
    for col in df.columns:
        if col != "deg-malig":
            df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    df["breast-quad"] = df["breast-quad"].apply(
        lambda v: v.replace("-", "_") if isinstance(v, str) else v
    )
    df["deg-malig"] = pd.to_numeric(df["deg-malig"], errors="coerce")
    return df[COLUNAS_UCI]
def explorar_dados(df: pd.DataFrame) -> None:
    """Mostra tamanho do dataset, valores ausentes e distribuição da classe alvo."""
    print("=" * 65)
    print("1. ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 65)
    print(f"Registros: {df.shape[0]} | Atributos + Alvo: {df.shape[1]}")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    print("\nValores ausentes ('?'):")
    if nulos.empty:
        print("  Nenhum valor ausente.")
    else:
        for col, qtd in nulos.items():
            print(f"  - {col}: {qtd}")
    print("\nDistribuição do atributo Class:")
    for classe, qtd in df["Class"].value_counts().items():
        print(f"  - {classe}: {qtd} ({(qtd / len(df)) * 100:.2f}%)")
def criar_preprocessador() -> ColumnTransformer:
    """Define o pré-processamento para atributos categóricos e numéricos."""
    # Imputa nulos pela moda e aplica One-Hot Encoding nas variáveis categóricas
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Imputa pela mediana e padroniza a variável numérica (deg-malig)
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, COLUNAS_CATEGORICAS),
            ("num", num_pipe, COLUNAS_NUMERICAS),
        ]
    )
def treinar_modelos(
    caminho_arquivo: Optional[Path] = None,
    caminho_modelo: Path = CAMINHO_MODELO,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Executa todo o pipeline de treinamento e seleciona o modelo com melhor acurácia.
    """
    df = carregar_dados(caminho_arquivo)
    explorar_dados(df)
    X = df[ATRIBUTOS]
    y = df["Class"]
    # Separa treino e teste antes do pré-processamento para evitar vazamento de dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessador = criar_preprocessador()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Modelos clássicos de classificação e hiperparâmetros para busca em grade
    modelos_candidatos = {
        "Regressão Logística": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 11]},
        ),
        "SVM": (
            SVC(random_state=42),
            {"clf__C": [0.5, 1.0, 2.0, 5.0], "clf__kernel": ["linear", "rbf"]},
        ),
        "Árvore de Decisão": (
            DecisionTreeClassifier(random_state=42),
            {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 6, 10, None]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
        ),
    }
    print("\n" + "=" * 65)
    print("2. TREINAMENTO, VALIDAÇÃO CRUZADA E SELEÇÃO POR ACURÁCIA")
    print("=" * 65)
    resultados = []
    pipelines_treinados = {}
    for nome, (estimador, grade) in modelos_candidatos.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("clf", estimador),
            ]
        )
        # GridSearchCV otimizando diretamente a acurácia em 5 folds estratificados
        busca = GridSearchCV(pipeline, grade, cv=cv, scoring="accuracy")
        busca.fit(X_train, y_train)
        melhor_pipe = busca.best_estimator_
        acc_teste = accuracy_score(y_test, melhor_pipe.predict(X_test))
        pipelines_treinados[nome] = melhor_pipe
        resultados.append(
            {
                "Modelo": nome,
                "Acuracia_CV": busca.best_score_,
                "Acuracia_Teste": acc_teste,
                "Parametros": busca.best_params_,
            }
        )
    # Seleciona o campeão pela maior acurácia média na validação cruzada
    df_resultados = (
        pd.DataFrame(resultados)
        .sort_values(by=["Acuracia_CV", "Acuracia_Teste"], ascending=False)
        .reset_index(drop=True)
    )
    print("\nRanking de desempenho dos modelos:")
    for _, linha in df_resultados.iterrows():
        print(
            f"  - {linha['Modelo']:<20} | "
            f"Acurácia CV: {linha['Acuracia_CV'] * 100:5.2f}% | "
            f"Acurácia Teste: {linha['Acuracia_Teste'] * 100:5.2f}%"
        )
    nome_campeao = df_resultados.loc[0, "Modelo"]
    melhor_modelo = pipelines_treinados[nome_campeao]
    y_pred = melhor_modelo.predict(X_test)
    print("\n" + "-" * 65)
    print(f"MODELO CAMPEÃO (MELHOR ACURÁCIA): {nome_campeao}")
    print(f"Acurácia média (CV 5-folds) : {df_resultados.loc[0, 'Acuracia_CV'] * 100:.2f}%")
    print(f"Acurácia no conjunto teste  : {df_resultados.loc[0, 'Acuracia_Teste'] * 100:.2f}%")
    print(f"Melhores hiperparâmetros    : {df_resultados.loc[0, 'Parametros']}")
    print("-" * 65)
    labels = ["no-recurrence-events", "recurrence-events"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nMatriz de Confusão (Teste):")
    print(pd.DataFrame(cm, index=[f"Real: {l}" for l in labels], columns=[f"Prev: {l}" for l in labels]))
    print("\nRelatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred, zero_division=0))
    # Salva o pipeline completo (pré-processamento + modelo) para uso na inferência
    joblib.dump(melhor_modelo, caminho_modelo)
    print(f"Modelo salvo em: {caminho_modelo}")
    return melhor_modelo, df_resultados
# =============================================================================
# MÓDULO DE INFERÊNCIA
# =============================================================================
def normalizar_paciente(dados_paciente: dict) -> pd.DataFrame:
    """Padroniza os dados da nova paciente para o formato esperado pelo pipeline."""
    registro = {}
    for col in ATRIBUTOS:
        valor = dados_paciente.get(col)
        if isinstance(valor, str):
            valor = valor.strip()
            if valor in ("", "?"):
                valor = None
            elif col == "breast-quad":
                # Aceita tanto hífen (site UCI) quanto underscore (arquivo .data)
                valor = valor.replace("-", "_")
        if col == "deg-malig":
            valor = int(valor) if valor is not None else None
        registro[col] = valor
    return pd.DataFrame([registro], columns=ATRIBUTOS)
def prever_diagnostico(
    dados_paciente: dict,
    modelo: Optional[Pipeline] = None,
    caminho_modelo: Path = CAMINHO_MODELO,
) -> dict:
    """
    Recebe os atributos de uma nova paciente e retorna o indicativo
    de diagnóstico (atributo Class).
    """
    if modelo is None:
        if not caminho_modelo.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{caminho_modelo}'. Execute o treinamento primeiro."
            )
        modelo = joblib.load(caminho_modelo)
    df_paciente = normalizar_paciente(dados_paciente)
    classe_prevista = str(modelo.predict(df_paciente)[0])
    descricoes = {
        "no-recurrence-events": "Sem indicativo de recorrência do câncer de mama",
        "recurrence-events": "Com indicativo de recorrência do câncer de mama",
    }
    resultado = {
        "Class": classe_prevista,
        "diagnostico": descricoes.get(classe_prevista, classe_prevista),
        "probabilidades": None,
    }
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(df_paciente)[0]
        resultado["probabilidades"] = {
            str(cls): round(float(prob) * 100, 2)
            for cls, prob in zip(modelo.classes_, probs)
        }
    return resultado
def coletar_paciente_terminal() -> dict:
    """Solicita os 9 atributos clínicos da nova paciente pelo terminal."""
    print("\nInforme os dados da nova paciente (ou '?' caso desconheça algum campo):")
    paciente = {}
    for col in ATRIBUTOS:
        opcoes = OPCOES_ATRIBUTOS[col]
        while True:
            entrada = input(f"  {col} {opcoes}: ").strip()
            entrada_norm = entrada.replace("-", "_") if col == "breast-quad" else entrada
            if entrada == "?" or entrada_norm in opcoes:
                paciente[col] = entrada_norm
                break
            print(f"    Entrada inválida para '{col}'. Use uma das opções listadas ou '?'.")
    return paciente
def exibir_resultado(resultado: dict) -> None:
    """Exibe de forma clara a classe prevista e o indicativo diagnóstico."""
    print("\n>>> RESULTADO DA INFERÊNCIA <<<")
    print(f"Atributo Class previsto : {resultado['Class']}")
    print(f"Indicativo diagnóstico  : {resultado['diagnostico']}")
    if resultado["probabilidades"]:
        probs_txt = " | ".join(
            f"{cls}: {pct:.2f}%" for cls, pct in resultado["probabilidades"].items()
        )
        print(f"Probabilidades estimadas: {probs_txt}")
# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
def main() -> None:
    # Executa o treinamento e obtém o modelo de melhor acurácia
    melhor_modelo, _ = treinar_modelos()
    # Demonstração do módulo de inferência com uma paciente exemplo
    print("\n" + "=" * 65)
    print("3. MÓDULO DE INFERÊNCIA - EXEMPLO DE DIAGNÓSTICO")
    print("=" * 65)
    paciente_exemplo = {
        "age": "40-49",
        "menopause": "premeno",
        "tumor-size": "30-34",
        "inv-nodes": "3-5",
        "node-caps": "yes",
        "deg-malig": 3,
        "breast": "left",
        "breast-quad": "left_low",
        "irradiat": "yes",
    }
    print("Paciente exemplo:", paciente_exemplo)
    resultado_exemplo = prever_diagnostico(paciente_exemplo, modelo=melhor_modelo)
    exibir_resultado(resultado_exemplo)
    # Loop opcional para o usuário testar novas pacientes manualmente
    while True:
        try:
            resp = input("\nDeseja testar uma nova paciente manualmente? (s/n): ").strip().lower()
        except EOFError:
            break
        if resp not in ("s", "sim"):
            break
        nova_paciente = coletar_paciente_terminal()
        resultado = prever_diagnostico(nova_paciente, modelo=melhor_modelo)
        exibir_resultado(resultado)
if __name__ == "__main__":
    main()