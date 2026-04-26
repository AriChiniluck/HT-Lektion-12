# Фінальний стан Langfuse трейсингу — HT Lektion 12

## Ієрархія трасування

```
run_traced_turn()          @observe(name="lecture12_user_turn")     ✅ main.py
│
├─ plan()                  @tool + @observe(name="plan_tool")        ✅ agents/planner.py  ← додано
│    └─ LLM (planner)      CallbackHandler()                         ✅
│
├─ research()              @tool + @observe(name="research_tool")    ✅ agents/research.py ← додано
│    └─ propagate_attributes(agent_name="researcher")                ✅ власний span
│    └─ LLM (researcher)   CallbackHandler()                         ✅
│         ├─ web_search()  @tool + @observe(name="web_search_tool")  ✅ tools.py
│         ├─ read_url()    @tool + @observe(name="read_url_tool")    ✅ tools.py
│         └─ knowledge_search() @observe(name="knowledge_search_tool") ✅ tools.py
│
├─ critique()              @tool + @observe(name="critique_tool")    ✅ agents/critic.py  ← додано
│    └─ propagate_attributes(agent_name="critic")                    ✅ власний span
│    └─ LLM (critic)       CallbackHandler()                         ✅
│
└─ save_report()           @tool + @observe(name="save_report_tool") ✅ tools.py
```

## Примітки

- **`@observe`** — обертає функцію у Langfuse span; вкладеність відстежується автоматично.
- **`propagate_attributes`** — передає `session_id`, `user_id`, `metadata` вниз по стеку (researcher та critic мають власний `agent_name` у метаданих).
- **`CallbackHandler()`** — `get_langfuse_handler()` передається як LangChain callback і відстежує LLM-токени, вартість та latency на рівні кожного агента окремо.
- **Чому Langfuse показує "not called" для деяких інструментів у конкретній generation**: кожна generation-нода відображає інструменти, доступні супервайзору, і лише ті, що були викликані *в тому конкретному LLM-кроці*. `plan`, `research`, `critique`, `save_report` — послідовні кроки у різних generation-нодах; розгорніть дерево щоб побачити кожен виклик.
- **`thread_id`** та **`session_id`** реєструються через `set_active_thread_id()` у `supervisor.py` перед кожним `supervisor.stream()` — безпечно для multi-user (threading.local()).
