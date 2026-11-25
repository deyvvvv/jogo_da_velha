from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jogo_velha import (
    ALFA_BETA,
    ALGORITMOS,
    MINIMAX,
    Metricas,
    aplicar_jogada,
    escolher_melhor_jogada,
    estado_terminal,
    novo_tabuleiro,
    outro,
    vencedor,
    X,
)

REPETICOES_TEMPO = 50
PREFERENCIA_HUMANA = (4, 0, 1, 2, 3, 5, 6, 7, 8)
CENARIOS = (
    ("computador_inicia", True, "Computador inicia (computador = X)"),
    ("humano_inicia", False, "Humano inicia (computador = O)"),
)
DIRETORIO_SAIDA = Path(__file__).resolve().parent / "resultados"

COLUNAS_CSV = [
    "cenario",
    "descricao_cenario",
    "algoritmo",
    "jogada_computador",
    "jogada_geral",
    "simbolo_computador",
    "tabuleiro",
    "jogada_escolhida",
    "valor_raiz",
    "nos",
    "transicoes",
    "cortes",
    "tempo_ms",
    "tempo_medio_ms",
]


def jogada_humana(tabuleiro: list[str]) -> int:
    """Política humana determinística, usada apenas para tornar o teste reprodutível."""
    for posicao in PREFERENCIA_HUMANA:
        if tabuleiro[posicao] == " ":
            return posicao
    raise ValueError("Não há jogada humana disponível.")


def simular_partida(algoritmo: str, computador_inicia: bool) -> dict[str, object]:
    """Roda uma partida completa e mede cada decisão do computador."""
    tabuleiro = novo_tabuleiro()
    simbolo_computador = X if computador_inicia else outro(X)
    simbolo_humano = outro(simbolo_computador)
    jogador_atual = X
    descricao = next(d for c, _, d in CENARIOS if (c == "computador_inicia") == computador_inicia)
    cenario = "computador_inicia" if computador_inicia else "humano_inicia"

    registros: list[dict[str, object]] = []
    jogada_computador = 0
    jogada_geral = 0

    while not estado_terminal(tabuleiro):
        jogada_geral += 1

        if jogador_atual == simbolo_humano:
            posicao = jogada_humana(tabuleiro)
            tabuleiro = aplicar_jogada(tabuleiro, posicao, simbolo_humano)
        else:
            jogada_computador += 1
            metricas = Metricas()
            posicao, valor = escolher_melhor_jogada(
                tabuleiro, simbolo_computador, algoritmo, metricas
            )

            # A jogada acima é a executada na partida. Repetimos a resolução do
            # mesmo estado apenas para reduzir o ruído da medição de tempo.
            amostras_ns: list[int] = []
            metricas_repeticao = Metricas()
            for _ in range(REPETICOES_TEMPO):
                escolher_melhor_jogada(
                    tabuleiro, simbolo_computador, algoritmo, metricas_repeticao
                )
                amostras_ns.append(metricas_repeticao.tempo_ns)

            registro = {
                "cenario": cenario,
                "descricao_cenario": descricao,
                "algoritmo": algoritmo,
                "jogada_computador": jogada_computador,
                "jogada_geral": jogada_geral,
                "simbolo_computador": simbolo_computador,
                "tabuleiro": "".join(tabuleiro).replace(" ", "."),
                "jogada_escolhida": posicao + 1,
                "valor_raiz": valor,
                "nos": metricas.nos,
                "transicoes": metricas.transicoes,
                "cortes": metricas.cortes,
                "tempo_ms": metricas.tempo_ns / 1_000_000,
                "tempo_medio_ms": statistics.mean(amostras_ns) / 1_000_000,
            }
            registros.append(registro)
            tabuleiro = aplicar_jogada(tabuleiro, posicao, simbolo_computador)

        jogador_atual = outro(jogador_atual)

    ganhador = vencedor(tabuleiro)
    return {
        "registros": registros,
        "estados_computador": [r["tabuleiro"] for r in registros],
        "jogadas_computador": [r["jogada_escolhida"] for r in registros],
        "resultado": ganhador if ganhador else "empate",
        "tabuleiro_final": "".join(tabuleiro).replace(" ", "."),
    }


def resume(registros: list[dict[str, object]]) -> dict[str, object]:
    """Agrega os custos por cenário e algoritmo."""
    return {
        "decisoes": len(registros),
        "nos_total": sum(int(r["nos"]) for r in registros),
        "transicoes_total": sum(int(r["transicoes"]) for r in registros),
        "cortes_total": sum(int(r["cortes"]) for r in registros),
        "tempo_total_ms": sum(float(r["tempo_medio_ms"]) for r in registros),
        "tempo_real_total_ms": sum(float(r["tempo_ms"]) for r in registros),
    }


def calcular_reducao(minimax: dict[str, object], alfabeta: dict[str, object]) -> dict[str, float]:
    """Redução percentual: (MinMax - alfa-beta) / MinMax."""
    chaves = ("nos_total", "transicoes_total", "tempo_total_ms")
    return {
        chave: 100.0 * (float(minimax[chave]) - float(alfabeta[chave])) / float(minimax[chave])
        for chave in chaves
    }


def gerar_graficos(registros: list[dict[str, object]], resumo: dict[str, object]) -> None:
    """Gera os gráficos usados na análise do relatório."""
    cores = {MINIMAX: "#2878B5", ALFA_BETA: "#E07B39"}
    rotulos = {MINIMAX: "MinMax", ALFA_BETA: "Alfa-beta"}
    nomes_cenario = {
        "computador_inicia": "Computador inicia (X)",
        "humano_inicia": "Humano inicia (computador O)",
    }

    for metrica, titulo, eixo, nome_arquivo in (
        ("nos", "Passos (nós examinados) por decisão", "Nós examinados", "passos_por_jogada.png"),
        (
            "tempo_medio_ms",
            "Tempo médio por decisão",
            "Tempo médio (ms)",
            "tempo_por_jogada.png",
        ),
    ):
        fig, eixos = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
        for eixo_grafico, (chave_cenario, _, _) in zip(eixos, CENARIOS):
            for algoritmo in ALGORITMOS:
                dados = [
                    r
                    for r in registros
                    if r["cenario"] == chave_cenario and r["algoritmo"] == algoritmo
                ]
                x = [int(r["jogada_computador"]) for r in dados]
                y = [float(r[metrica]) for r in dados]
                eixo_grafico.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    color=cores[algoritmo],
                    label=rotulos[algoritmo],
                )
            eixo_grafico.set_title(nomes_cenario[chave_cenario])
            eixo_grafico.set_xlabel("Jogada do computador")
            eixo_grafico.set_ylabel(eixo)
            eixo_grafico.grid(alpha=0.3)
            eixo_grafico.legend()
        fig.suptitle(titulo, fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(DIRETORIO_SAIDA / nome_arquivo, dpi=180)
        plt.close(fig)

    # Custos totais por partida.
    chaves_grafico = (
        ("nos_total", "Nós examinados"),
        ("transicoes_total", "Transições geradas"),
        ("tempo_total_ms", "Tempo médio total (ms)"),
    )
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6))
    for eixo_grafico, (chave, titulo_eixo) in zip(eixos, chaves_grafico):
        rotulos_x = [nomes_cenario[c[0]] for c in CENARIOS]
        posicoes = range(len(rotulos_x))
        largura = 0.35
        valores_mm = [resumo["resumo"][c[0]][MINIMAX][chave] for c in CENARIOS]
        valores_ab = [resumo["resumo"][c[0]][ALFA_BETA][chave] for c in CENARIOS]
        eixo_grafico.bar(
            [p - largura / 2 for p in posicoes],
            valores_mm,
            largura,
            label="MinMax",
            color=cores[MINIMAX],
        )
        eixo_grafico.bar(
            [p + largura / 2 for p in posicoes],
            valores_ab,
            largura,
            label="Alfa-beta",
            color=cores[ALFA_BETA],
        )
        eixo_grafico.set_xticks(list(posicoes))
        eixo_grafico.set_xticklabels(["Computador inicia", "Humano inicia"])
        eixo_grafico.set_ylabel(titulo_eixo)
        eixo_grafico.set_title(titulo_eixo)
        eixo_grafico.grid(axis="y", alpha=0.3)
        eixo_grafico.legend()
        for indice, valor in enumerate(valores_mm):
            eixo_grafico.text(
                indice - largura / 2, valor, f"{valor:,.0f}",
                ha="center", va="bottom", fontsize=8,
            )
        for indice, valor in enumerate(valores_ab):
            eixo_grafico.text(
                indice + largura / 2, valor, f"{valor:,.0f}",
                ha="center", va="bottom", fontsize=8,
            )
    fig.suptitle("Custo total das duas partidas", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DIRETORIO_SAIDA / "custo_total.png", dpi=180)
    plt.close(fig)

    # Redução percentual proporcionada pela poda.
    fig, eixo_grafico = plt.subplots(figsize=(9, 5))
    rotulos_x = ["Nós", "Transições", "Tempo"]
    chaves_reducao = ("nos_total", "transicoes_total", "tempo_total_ms")
    posicoes = range(len(rotulos_x))
    largura = 0.35
    for deslocamento, (chave_cenario, _, _) in zip((-largura / 2, largura / 2), CENARIOS):
        reducoes = [
            resumo["reducao"][chave_cenario][chave] for chave in chaves_reducao
        ]
        barras = eixo_grafico.bar(
            [p + deslocamento for p in posicoes],
            reducoes,
            largura,
            label=nomes_cenario[chave_cenario],
        )
        for barra, valor in zip(barras, reducoes):
            eixo_grafico.text(
                barra.get_x() + barra.get_width() / 2,
                valor,
                f"{valor:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    eixo_grafico.set_xticks(list(posicoes))
    eixo_grafico.set_xticklabels(rotulos_x)
    eixo_grafico.set_ylabel("Redução em relação ao MinMax (%)")
    eixo_grafico.set_title("Economia obtida com a poda alfa-beta")
    eixo_grafico.grid(axis="y", alpha=0.3)
    eixo_grafico.legend()
    fig.tight_layout()
    fig.savefig(DIRETORIO_SAIDA / "reducao_alfa_beta.png", dpi=180)
    plt.close(fig)


def main() -> None:
    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)

    todos_registros: list[dict[str, object]] = []
    resumo: dict[str, object] = {
        "configuracao": {
            "repeticoes_para_tempo": REPETICOES_TEMPO,
            "preferencia_humana": list(PREFERENCIA_HUMANA),
            "ordem_busca": [4, 0, 2, 6, 8, 1, 3, 5, 7],
            "definicao_passo": "uma chamada recursiva da busca (um nó examinado)",
        },
        "partidas": {},
        "resumo": {},
        "reducao": {},
    }

    print("Executando experimento Humano x Computador...")
    for chave_cenario, computador_inicia, descricao in CENARIOS:
        partidas = {
            algoritmo: simular_partida(algoritmo, computador_inicia)
            for algoritmo in ALGORITMOS
        }

        # Garantia de que os algoritmos resolveram exatamente os mesmos estados.
        estados_mm = partidas[MINIMAX]["estados_computador"]
        estados_ab = partidas[ALFA_BETA]["estados_computador"]
        jogadas_mm = partidas[MINIMAX]["jogadas_computador"]
        jogadas_ab = partidas[ALFA_BETA]["jogadas_computador"]
        if estados_mm != estados_ab or jogadas_mm != jogadas_ab:
            raise RuntimeError(
                "MinMax e alfa-beta não chegaram aos mesmos estados/jogadas. "
                "A comparação não seria justa."
            )

        registros = partidas[MINIMAX]["registros"] + partidas[ALFA_BETA]["registros"]
        todos_registros.extend(registros)

        resumo["partidas"][chave_cenario] = {
            "descricao": descricao,
            "resultado": partidas[MINIMAX]["resultado"],
            "tabuleiro_final": partidas[MINIMAX]["tabuleiro_final"],
            "estados_computador": estados_mm,
            "jogadas_computador": jogadas_mm,
        }
        resumo["resumo"][chave_cenario] = {
            MINIMAX: resume(partidas[MINIMAX]["registros"]),
            ALFA_BETA: resume(partidas[ALFA_BETA]["registros"]),
        }
        resumo["reducao"][chave_cenario] = calcular_reducao(
            resumo["resumo"][chave_cenario][MINIMAX],
            resumo["resumo"][chave_cenario][ALFA_BETA],
        )

        print(f"  {descricao}: resultado = {partidas[MINIMAX]['resultado']}")

    with (DIRETORIO_SAIDA / "metricas.csv").open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_CSV)
        escritor.writeheader()
        escritor.writerows(todos_registros)

    with (DIRETORIO_SAIDA / "resumo.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    gerar_graficos(todos_registros, resumo)

    print(f"\nArquivos gerados em: {DIRETORIO_SAIDA}")
    for chave_cenario, _, descricao in CENARIOS:
        mm = resumo["resumo"][chave_cenario][MINIMAX]
        ab = resumo["resumo"][chave_cenario][ALFA_BETA]
        reducao = resumo["reducao"][chave_cenario]
        print(f"\n{descricao}")
        print(
            f"  MinMax:   {mm['nos_total']} nós, {mm['transicoes_total']} transições, "
            f"{mm['tempo_total_ms']:.4f} ms"
        )
        print(
            f"  Alfa-beta:{ab['nos_total']:>6} nós, {ab['transicoes_total']:>6} transições, "
            f"{ab['tempo_total_ms']:.4f} ms"
        )
        print(
            f"  Redução:  {reducao['nos_total']:.1f}% nós, "
            f"{reducao['transicoes_total']:.1f}% transições, "
            f"{reducao['tempo_total_ms']:.1f}% tempo"
        )


if __name__ == "__main__":
    main()
