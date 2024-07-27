"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time

import gradio as gr

from fastchat.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from fastchat.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from fastchat.serve.gradio_block_arena_vision import (
    build_single_vision_language_model_ui,
)
from fastchat.serve.gradio_block_arena_vision_anony import (
    build_side_by_side_vision_ui_anony,
    load_demo_side_by_side_vision_anony,
)
from fastchat.serve.gradio_block_arena_vision_named import (
    build_side_by_side_vision_ui_named,
)

from fastchat.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_about,
    get_model_list,
    load_demo_single,
    get_ip,
)
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    alert_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(url_params, request: gr.Request):
    global models, all_models, vl_models, all_vl_models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")

    inner_selected = 0
    if "arena" in url_params:
        inner_selected = 0
    elif "vision" in url_params:
        inner_selected = 1
    elif "compare" in url_params:
        inner_selected = 1
    elif "direct" in url_params or "model" in url_params:
        inner_selected = 3
    elif "leaderboard" in url_params:
        inner_selected = 4
    elif "about" in url_params:
        inner_selected = 5

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=False,
        )

        vl_models, all_vl_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=True,
        )

    single_updates = load_demo_single(models, url_params)
    side_by_side_anony_updates = load_demo_side_by_side_anony(all_models, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)

    side_by_side_vision_anony_updates = load_demo_side_by_side_vision_anony(
        all_models, all_vl_models, url_params
    )

    return (
        (gr.Tabs(selected=inner_selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_named_updates
        + side_by_side_vision_anony_updates
    )


def build_demo(models, vl_models, elo_results_file, leaderboard_table_file):
    if args.show_terms_of_use:
        load_js = get_window_url_params_with_tos_js
    else:
        load_js = get_window_url_params_js

    head_js = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    """
    if args.ga_id is not None:
        head_js += f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={args.ga_id}"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());

    gtag('config', '{args.ga_id}');
    window.__gradio_mode__ = "app";
    </script>
    """
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="与开放大语言模型对话",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
        head=head_js,
    ) as demo:
        with gr.Tabs() as inner_tabs:
            if args.vision_arena:
                with gr.Tab("⚔️ 视觉竞技场", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_vision_ui_anony(
                        all_models,
                        all_vl_models,
                        random_questions=args.random_questions,
                    )
            else:
                with gr.Tab("⚔️ 语言竞技场", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_ui_anony(models)

            with gr.Tab("🔍 模型对比", id=2) as side_by_side_tab:
                side_by_side_tab.select(None, None, None, js=alert_js)
                side_by_side_named_list = build_side_by_side_ui_named(models)

            with gr.Tab("💬 直接对话", id=3) as direct_tab:
                direct_tab.select(None, None, None, js=alert_js)
                single_model_list = build_single_model_ui(
                    models, add_promotion_links=True
                )

            demo_tabs = (
                [inner_tabs]
                + single_model_list
                + side_by_side_anony_list
                + side_by_side_named_list
            )

            if elo_results_file:
                with gr.Tab("🏆 排行榜", id=4):
                    build_leaderboard_tab(
                        elo_results_file, leaderboard_table_file, show_plot=True
                    )

            with gr.Tab("ℹ️ 关于我们", id=5):
                about = build_about()

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"未知的模型列表模式: {args.model_list_mode}")

        demo.load(
            load_demo,
            [url_params],
            demo_tabs,
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, help="服务器端口号")
    parser.add_argument(
        "--share",
        action="store_true",
        help="是否生成一个公开的可分享链接",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="控制器的地址",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="Gradio队列的并发数",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="是否只加载一次模型列表或每次都重新加载",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="启用内容审核以阻止不安全的输入",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="在加载演示之前显示使用条款",
    )
    parser.add_argument(
        "--vision-arena", 
        action="store_true", 
        help="显示视觉竞技场的标签页"
    )
    parser.add_argument(
        "--random-questions", 
        type=str, 
        help="从JSON文件加载随机问题"
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="从JSON文件注册基于API的模型端点",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='设置Gradio认证文件路径。文件应包含一个或多个用户:密码对，格式为："用户1:密码1,用户2:密码2,用户3:密码3"',
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", 
        type=str, 
        help="加载ELO排行榜结果和图表"
    )
    parser.add_argument(
        "--leaderboard-table-file", 
        type=str, 
        help="加载排行榜结果和图表"
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="设置Gradio根路径，例如 /abc/def。在反向代理后面运行或在自定义URL路径前缀下运行时很有用",
    )
    parser.add_argument(
        "--ga-id",
        type=str,
        help="Google Analytics ID",
        default=None,
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="如果设置为true，则将图像文件上传到Google Cloud Storage",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="设置Gradio网页服务器的密码",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    models, all_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=False,
    )

    vl_models, all_vl_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=True,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(
        models,
        all_vl_models,
        args.elo_results_file,
        args.leaderboard_table_file,
    )
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
        show_api=False,
    )
