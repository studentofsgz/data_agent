import asyncio

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


async def main():
    QUESTION = "各品牌的GMV排名"

    # 初始化
    embedding_client_manager.init()
    qdrant_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()

    async with (
        meta_mysql_client_manager.session_factory() as meta_session,
        dw_mysql_client_manager.session_factory() as dw_session,
    ):
        context = DataAgentContext(
            embedding_client=embedding_client_manager.client,
            column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
            value_es_repository=ValueESRepository(es_client_manager.client),
            metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
            meta_mysql_repository=MetaMySQLRepository(meta_session),
            dw_mysql_repository=DWMySQLRepository(dw_session),
        )

        state = DataAgentState(query=QUESTION)

        async for chunk in graph.astream(input=state, context=context, stream_mode="custom"):
            t = chunk.get("type", "")

            if t == "progress":
                step = chunk.get("step", "?")
                status = chunk.get("status", "?")
                print(f"[{status:7s}] {step}")

            elif t == "sql_preview":
                print(f"\n📝 生成的SQL:\n{chunk['sql']}\n")

            elif t == "result":
                data = chunk.get("data", [])
                print(f"📊 结果: {len(data)} 行")
                if data:
                    for row in data[:5]:
                        print(f"   {row}")

            elif t == "error":
                print(f"❌ 错误: {chunk.get('message', '未知')}")

    # 关闭
    await qdrant_client_manager.close()
    await es_client_manager.close()
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
