"""Original project source identities, not deployment paths or model secrets."""
ORIGINAL_SHA256={
    'run_model.py':'bca323d19dfe11bcf0d04e0d4800d120f85605b648f85ff6ab52b1d26fa6d3fe',
    'codex_client/app_server.py':'8c16efa84bba8b497ace99e46ccd1c58a3f29feb39fed40b7239341737b0c29d',
    'codex_client/session.py':'09d7dac7ed090cf10bf03b81857463263815e89afc8bbca95f80a6dc458adca7',
    'serve_codex.py':'eb19a26f8a862442dbbb8168bfb4f33a508e2cdc9b726da50189a344d6d9b39d',
    'native_tools/codex_adapter_v1/bridge.py':'5362ec305aa1725c9be8190b8a007ee89250287e5dadec330159023e4a47bb77',
    'native_tools/contract.py':'84283d55a3c7644b6c4a4e22bff5971d27ccd1505c0bc867165bda59c93f7d93',
}
ORIGINAL_RECIPE=dict(model='gpt-6-astra',effort='medium',max_model_responses=200,
    authentication='user-owned ChatGPT login',provider='openai',transport='original_default',
    api_fallback=False,control_hz=20,raw_action_dimensions=22)
WIRE='experimental dynamicTools namespace arena; JSONL stdio://; item/tool/call'
