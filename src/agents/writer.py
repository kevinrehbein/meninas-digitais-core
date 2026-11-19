from langchain_core.prompts import ChatPromptTemplate

from config import WRITER_TEMPERATURE

from .llm import get_llm
from .prompts import WRITER_PROMPT
from .state import AgentState
from .utils import (
    build_revision_alert,
    get_research_evidence,
    get_user_question,
    message_text,
)

_INSTRUCTIONS = {
    "qa": "Responda exatamente o que foi perguntado. Se a pergunta pedir uma listagem, liste todos os itens relevantes, consolidados; caso contrário, seja direto.",
    "sintese": "Escreva uma síntese acadêmica separada por seções (Markdown), cobrindo todos os itens relevantes consolidados e citando as fontes.",
}


def redator_node(state: AgentState) -> dict:
    """Responde perguntas diretas (QA) ou cria sínteses; reescreve se houver feedback."""
    instrucao = _INSTRUCTIONS.get(state.get("intent") or "qa", _INSTRUCTIONS["sintese"])
    instrucao += build_revision_alert(state.get("feedback"), "texto")

    prompt = ChatPromptTemplate.from_messages(
        [("system", WRITER_PROMPT), ("user", "Pergunta do usuário: {input}")]
    )
    pipe = prompt | get_llm(temperature=WRITER_TEMPERATURE)
    resposta = pipe.invoke(
        {
            "contexto": get_research_evidence(state),
            "instrucao": instrucao,
            "input": get_user_question(state),
        }
    )
    return {"draft": message_text(resposta)}
