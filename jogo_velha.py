from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Optional

X = "X"
O = "O"
VAZIO = " "

LINHAS_VITORIA = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

# Ordem fixa: centro, cantos, laterais. Ela é usada pelos dois algoritmos,
# o que torna a comparação justa e reproduzível.
ORDEM_BUSCA = (4, 0, 2, 6, 8, 1, 3, 5, 7)

MINIMAX = "minimax"
ALFA_BETA = "alfa-beta"
ALGORITMOS = (MINIMAX, ALFA_BETA)

Tabuleiro = list[str]


@dataclass
class Metricas:
    """Contadores coletados em uma única decisão do computador."""

    nos: int = 0
    transicoes: int = 0
    cortes: int = 0
    tempo_ns: int = 0
    melhor_valor: Optional[float] = None
    jogada: Optional[int] = None

    def zerar(self) -> None:
        self.nos = 0
        self.transicoes = 0
        self.cortes = 0
        self.tempo_ns = 0
        self.melhor_valor = None
        self.jogada = None

    def como_dicionario(self) -> dict[str, object]:
        return {
            "nos": self.nos,
            "transicoes": self.transicoes,
            "cortes": self.cortes,
            "tempo_ms": self.tempo_ns / 1_000_000,
            "melhor_valor": self.melhor_valor,
            "jogada": self.jogada,
        }


def outro(jogador: str) -> str:
    """Devolve o símbolo do adversário."""
    if jogador not in (X, O):
        raise ValueError(f"Jogador inválido: {jogador!r}")
    return O if jogador == X else X


def novo_tabuleiro() -> Tabuleiro:
    """Cria um tabuleiro vazio. A posição 0 é o canto superior esquerdo."""
    return [VAZIO] * 9


def vencedor(tabuleiro: Tabuleiro) -> Optional[str]:
    """Retorna X, O ou None, conforme o estado terminal."""
    for a, b, c in LINHAS_VITORIA:
        if tabuleiro[a] != VAZIO and tabuleiro[a] == tabuleiro[b] == tabuleiro[c]:
            return tabuleiro[a]
    return None


def tabuleiro_cheio(tabuleiro: Tabuleiro) -> bool:
    return VAZIO not in tabuleiro


def estado_terminal(tabuleiro: Tabuleiro) -> bool:
    return vencedor(tabuleiro) is not None or tabuleiro_cheio(tabuleiro)


def utilidade(tabuleiro: Tabuleiro, jogador_referencia: str) -> int:
    """Avaliação exigida no enunciado: +1 vitória, -1 derrota, 0 empate.

    A avaliação é sempre relativa a ``jogador_referencia``, ou seja, ao
    jogador que fez a Raiz da busca. Assim funciona tanto quando o
    computador joga de X quanto quando joga de O.
    """
    ganhador = vencedor(tabuleiro)
    if ganhador is None:
        return 0
    return 1 if ganhador == jogador_referencia else -1


def jogadas_possiveis(tabuleiro: Tabuleiro) -> list[int]:
    return [i for i in ORDEM_BUSCA if tabuleiro[i] == VAZIO]


def aplicar_jogada(tabuleiro: Tabuleiro, posicao: int, jogador: str) -> Tabuleiro:
    if posicao < 0 or posicao > 8:
        raise ValueError("A posição deve estar entre 0 e 8.")
    if tabuleiro[posicao] != VAZIO:
        raise ValueError("Essa posição já está ocupada.")
    filho = tabuleiro.copy()
    filho[posicao] = jogador
    return filho


def minimax(
    tabuleiro: Tabuleiro,
    jogador_atual: str,
    jogador_referencia: str,
    maximizar: bool,
    metricas: Metricas,
) -> int:
    """MinMax clássico. Cada chamada corresponde a um nó/passo examinado."""
    metricas.nos += 1

    if estado_terminal(tabuleiro):
        return utilidade(tabuleiro, jogador_referencia)

    jogadas = jogadas_possiveis(tabuleiro)
    if maximizar:
        melhor = float("-inf")
        for jogada in jogadas:
            metricas.transicoes += 1
            filho = aplicar_jogada(tabuleiro, jogada, jogador_atual)
            valor = minimax(
                filho,
                outro(jogador_atual),
                jogador_referencia,
                False,
                metricas,
            )
            melhor = max(melhor, valor)
        return int(melhor)

    melhor = float("inf")
    for jogada in jogadas:
        metricas.transicoes += 1
        filho = aplicar_jogada(tabuleiro, jogada, jogador_atual)
        valor = minimax(
            filho,
            outro(jogador_atual),
            jogador_referencia,
            True,
            metricas,
        )
        melhor = min(melhor, valor)
    return int(melhor)


def alfa_beta(
    tabuleiro: Tabuleiro,
    jogador_atual: str,
    jogador_referencia: str,
    maximizar: bool,
    alfa: float,
    beta: float,
    metricas: Metricas,
) -> int:
    """MinMax com poda alfa-beta; não altera o valor retornado pelo MinMax."""
    metricas.nos += 1

    if estado_terminal(tabuleiro):
        return utilidade(tabuleiro, jogador_referencia)

    jogadas = jogadas_possiveis(tabuleiro)
    if maximizar:
        melhor = float("-inf")
        for jogada in jogadas:
            metricas.transicoes += 1
            filho = aplicar_jogada(tabuleiro, jogada, jogador_atual)
            valor = alfa_beta(
                filho,
                outro(jogador_atual),
                jogador_referencia,
                False,
                alfa,
                beta,
                metricas,
            )
            melhor = max(melhor, valor)
            alfa = max(alfa, melhor)
            if alfa >= beta:
                metricas.cortes += 1
                break
        return int(melhor)

    melhor = float("inf")
    for jogada in jogadas:
        metricas.transicoes += 1
        filho = aplicar_jogada(tabuleiro, jogada, jogador_atual)
        valor = alfa_beta(
            filho,
            outro(jogador_atual),
            jogador_referencia,
            True,
            alfa,
            beta,
            metricas,
        )
        melhor = min(melhor, valor)
        beta = min(beta, melhor)
        if beta <= alfa:
            metricas.cortes += 1
            break
    return int(melhor)


def escolher_melhor_jogada(
    tabuleiro: Tabuleiro,
    jogador: str,
    algoritmo: str,
    metricas: Optional[Metricas] = None,
) -> tuple[int, int]:
    """Escolhe a jogada ótima e preenche ``metricas`` com o custo da busca.

    A desempate é determinístico: quando duas jogadas têm o mesmo valor,
    prevalece a primeira na ``ORDEM_BUSCA``. MinMax e alfa-beta, portanto,
    escolhem exatamente a mesma jogada.
    """
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo!r}")
    if estado_terminal(tabuleiro):
        raise ValueError("Não há jogada: o tabuleiro já é terminal.")

    metricas = metricas or Metricas()
    metricas.zerar()

    inicio = perf_counter_ns()
    metricas.nos = 1

    melhor_valor = float("-inf")
    melhor_jogada: Optional[int] = None

    if algoritmo == MINIMAX:
        for jogada in jogadas_possiveis(tabuleiro):
            metricas.transicoes += 1
            filho = aplicar_jogada(tabuleiro, jogada, jogador)
            valor = minimax(
                filho,
                outro(jogador),
                jogador,
                False,
                metricas,
            )
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_jogada = jogada
    else:
        alfa = float("-inf")
        for jogada in jogadas_possiveis(tabuleiro):
            metricas.transicoes += 1
            filho = aplicar_jogada(tabuleiro, jogada, jogador)
            valor = alfa_beta(
                filho,
                outro(jogador),
                jogador,
                False,
                alfa,
                float("inf"),
                metricas,
            )
            if valor > melhor_valor:
                melhor_valor = valor
                melhor_jogada = jogada
            alfa = max(alfa, melhor_valor)

    metricas.tempo_ns = perf_counter_ns() - inicio
    metricas.melhor_valor = int(melhor_valor)
    assert melhor_jogada is not None
    metricas.jogada = melhor_jogada
    return melhor_jogada, int(melhor_valor)


def mostrar_tabuleiro(tabuleiro: Tabuleiro) -> None:
    print()
    for linha in range(3):
        celulas = [tabuleiro[linha * 3 + coluna] or "-" for coluna in range(3)]
        print(f" {celulas[0]} | {celulas[1]} | {celulas[2]} ")
        if linha < 2:
            print("---+---+---")
    print("Posições: 1 2 3 / 4 5 6 / 7 8 9\n")


def ler_opcao(mensagem: str, opcoes: tuple[str, ...]) -> str:
    while True:
        resposta = input(mensagem).strip().upper()
        if resposta in opcoes:
            return resposta
        print(f"Opção inválida. Use uma destas: {', '.join(opcoes)}.")


def escolher_algoritmo(rotulo: str) -> str:
    print(f"\nAlgoritmo para {rotulo}:")
    print(" 1 - MinMax")
    print(" 2 - MinMax com poda alfa-beta")
    return MINIMAX if ler_opcao("Escolha: ", ("1", "2")) == "1" else ALFA_BETA


def jogar() -> None:
    """Interface interativa solicitada no enunciado."""
    print("=== Jogo da Velha — Inteligência Artificial ===")
    print("Modos de funcionamento:")
    print(" 1 - Humano x Computador")
    print(" 2 - Computador x Computador")
    modo = ler_opcao("Escolha o modo (1 ou 2): ", ("1", "2"))

    print("\nQuem iniciará a partida?")
    print(" 1 - Humano (joga com X)")
    print(" 2 - Computador (joga com X)")
    inicia_computador = ler_opcao("Escolha (1 ou 2): ", ("1", "2")) == "2"

    if modo == "1":
        algoritmo = escolher_algoritmo("o computador")
        algoritmos = {X: algoritmo, O: algoritmo}
    else:
        algoritmos = {
            X: escolher_algoritmo("o computador que joga com X"),
            O: escolher_algoritmo("o computador que joga com O"),
        }

    tabuleiro = novo_tabuleiro()
    jogador_atual = X
    humano = None if modo == "2" else (O if inicia_computador else X)
    total_nos = 0
    total_transicoes = 0
    total_cortes = 0
    total_tempo_ns = 0
    decisoes = 0

    while not estado_terminal(tabuleiro):
        mostrar_tabuleiro(tabuleiro)
        print(f"Vez de {jogador_atual}.")

        if jogador_atual == humano:
            while True:
                entrada = input("Escolha uma posição de 1 a 9: ").strip()
                if entrada.isdigit() and 1 <= int(entrada) <= 9:
                    posicao = int(entrada) - 1
                    if tabuleiro[posicao] == VAZIO:
                        break
                print("Jogada inválida. Informe uma posição livre de 1 a 9.")
            tabuleiro[posicao] = humano
        else:
            algoritmo = algoritmos[jogador_atual]
            metricas = Metricas()
            posicao, valor = escolher_melhor_jogada(
                tabuleiro, jogador_atual, algoritmo, metricas
            )
            tabuleiro[posicao] = jogador_atual
            print(
                f"Computador ({jogador_atual}, {algoritmo}) jogou na posição "
                f"{posicao + 1}. Valor da raiz: {valor:+d}"
            )
            print(
                f"Custo desta decisão: {metricas.nos} nós, "
                f"{metricas.transicoes} transições, {metricas.cortes} cortes, "
                f"{metricas.tempo_ns / 1_000_000:.4f} ms"
            )
            total_nos += metricas.nos
            total_transicoes += metricas.transicoes
            total_cortes += metricas.cortes
            total_tempo_ns += metricas.tempo_ns
            decisoes += 1

        jogador_atual = outro(jogador_atual)

    mostrar_tabuleiro(tabuleiro)
    ganhador = vencedor(tabuleiro)
    if ganhador is None:
        print("Fim de jogo: empate.")
    else:
        print(f"Fim de jogo: {ganhador} venceu!")

    if decisoes:
        print("\n--- Resumo das decisões do computador ---")
        print(f"Decisões: {decisoes}")
        print(f"Nós examinados: {total_nos}")
        print(f"Transições geradas: {total_transicoes}")
        print(f"Cortes alfa-beta: {total_cortes}")
        print(f"Tempo total de busca: {total_tempo_ns / 1_000_000:.4f} ms")


if __name__ == "__main__":
    jogar()
