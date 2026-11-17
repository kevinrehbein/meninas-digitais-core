import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from neo4j import GraphDatabase

LABEL = "__Entity__"
DRY_RUN = False      # True = só lista os grupos; False = funde
POR_TIPO = False    # True = só funde nós do mesmo tipo (Projeto com Projeto etc.)

SIGLAS_LEGIVEIS = {
    "SBC": "Sociedade Brasileira de Computação",
    "TI": "Tecnologia da Informação",
    "PMD": "Meninas Digitais",
    "WIT": "Women in Information Technology",
}

PREFIXO_RE = re.compile(
    r"^(area|local|projeto|programa|acao|conceito)(\s+(de|do|da|dos|das))?\s+"
)


def sem_acento(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def normalizar(nome: str) -> str:
    nome = re.sub(r"\([^)]*\)", " ", nome)  # remove "(SBC)" e similares
    return sem_acento(nome.replace("_", " ")).lower().strip()


def compactar(texto: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", texto)


SIGLAS = {
    compactar(normalizar(sigla)): compactar(normalizar(completo))
    for sigla, completo in SIGLAS_LEGIVEIS.items()
}


def chave(nome: str) -> str:
    n = normalizar(nome)
    sem_prefixo = PREFIXO_RE.sub("", n, count=1)
    if sem_prefixo:  # nunca deixa a chave vazia
        n = sem_prefixo
    n = compactar(n)
    return SIGLAS.get(n, n)


def melhor_nome(nome: str) -> tuple:
    return (
        not PREFIXO_RE.match(normalizar(nome)),  # prefere sem prefixo
        "_" not in nome,
        " " in nome,
        nome != sem_acento(nome),                # prefere com acento
        nome != nome.lower(),
    )


def tipo_do_no(rotulos: list[str]) -> str:
    return next((r for r in rotulos if r != LABEL), "")


def buscar_grupos(session) -> list[list[dict]]:
    nos = session.run(
        f"""
        MATCH (e:{LABEL}) WHERE e.id IS NOT NULL
        RETURN elementId(e) AS id, e.id AS nome, labels(e) AS rotulos,
               count {{ (e)--() }} AS grau
        """
    ).data()

    grupos = defaultdict(list)
    for no in nos:
        k = chave(no["nome"])
        if not k:
            continue
        if POR_TIPO:
            k = (tipo_do_no(no["rotulos"]), k)
        grupos[k].append(no)

    return [g for g in grupos.values() if len(g) > 1]


def unir_grupo(session, grupo: list[dict]) -> None:
    grupo.sort(key=lambda n: (melhor_nome(n["nome"]), n["grau"]), reverse=True)
    principal, outros = grupo[0], grupo[1:]
    aliases = sorted({n["nome"] for n in grupo})

    session.run(
        f"""
        MATCH (p:{LABEL}) WHERE elementId(p) = $id_principal
        MATCH (o:{LABEL}) WHERE elementId(o) IN $ids_outros
        WITH p, collect(o) AS outros
        CALL apoc.refactor.mergeNodes([p] + outros, {{
            properties: {{ `.*`: 'discard' }},
            mergeRels: true,
            produceSelfRel: false
        }}) YIELD node
        SET node.aliases = $aliases
        """,
        id_principal=principal["id"],
        ids_outros=[n["id"] for n in outros],
        aliases=aliases,
    )


def descrever(grupo: list[dict]) -> str:
    itens = sorted({(tipo_do_no(n["rotulos"]), n["nome"]) for n in grupo})
    mistura = len({t for t, _ in itens}) > 1
    texto = " | ".join(f"{nome} [{tipo}]" for tipo, nome in itens)
    return ("(TIPOS MISTOS) " if mistura else "") + texto


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            grupos = buscar_grupos(session)
            grupos.sort(key=len, reverse=True)

            print(f"Grupos de duplicatas: {len(grupos)}")
            print(f"Nós removidos pelo merge: {sum(len(g) - 1 for g in grupos)}")

            for g in grupos:
                print(descrever(g))

            if DRY_RUN:
                print("\nDRY_RUN ativo: nada foi alterado.")
                return

            for i, g in enumerate(grupos, start=1):
                unir_grupo(session, g)
                if i % 100 == 0:
                    print(f"  {i}/{len(grupos)} grupos unidos...")
            print("\nConcluído.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()