"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    TEXT_MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
    disable_text,
    enable_text,
)
from fastchat.serve.gradio_block_arena_anony import (
    flash_buttons,
    vote_last_response,
    leftvote_last_response,
    rightvote_last_response,
    tievote_last_response,
    bothbad_vote_last_response,
    regenerate,
    clear_history,
    share_click,
    add_text,
    bot_response_multi,
    set_global_vars_anony,
    load_demo_side_by_side_anony,
    get_sample_weight,
    get_battle_pair,
    SAMPLING_WEIGHTS,
    BATTLE_TARGETS,
    SAMPLING_BOOST_MODELS,
    OUTAGE_MODELS,
)
from fastchat.serve.gradio_block_arena_vision import (
    set_invisible_image,
    set_visible_image,
    add_image,
    moderate_input,
    enable_multimodal,
    _prepare_text_with_image,
    convert_images_to_conversation_format,
    invisible_text,
    visible_text,
    disable_multimodal,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
text_models = []
vl_models = []

# TODO(chris): fix sampling weights
VISION_SAMPLING_WEIGHTS = {
    "gpt-4o-2024-05-13": 4,
    "gpt-4-turbo-2024-04-09": 4,
    "claude-3-haiku-20240307": 4,
    "claude-3-sonnet-20240229": 4,
    "claude-3-5-sonnet-20240620": 4,
    "claude-3-opus-20240229": 4,
    "gemini-1.5-flash-api-0514": 4,
    "gemini-1.5-pro-api-0514": 4,
    "llava-v1.6-34b": 4,
    "reka-core-20240501": 4,
    "reka-flash-preview-20240611": 4,
}

# TODO(chris): Find battle targets that make sense
VISION_BATTLE_TARGETS = {}

# TODO(chris): Fill out models that require sampling boost
VISION_SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
VISION_OUTAGE_MODELS = []


def get_vqa_sample():
    random_sample = np.random.choice(vqa_samples)
    question, path = random_sample["question"], random_sample["path"]
    res = {"text": "", "files": [path]}
    return (res, path)


def load_demo_side_by_side_vision_anony(all_text_models, all_vl_models, url_params):
    global text_models, vl_models
    text_models = all_text_models
    vl_models = all_vl_models

    states = (None,) * num_sides
    selector_updates = (
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    )

    return states + selector_updates


def clear_history_example(request: gr.Request):
    logger.info(f"clear_history_example (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_multimodal, invisible_text]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [enable_btn]
    )


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    filename = get_conv_log_filename(states[0].is_vision, states[0].has_csam_image)

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

    gr.Info("🎉 感谢您的投票! 您的投票将影响排行榜,请负责任地投票。")
    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            yield names + (disable_text,) + (disable_btn,) * 4
            time.sleep(0.1)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        yield names + (disable_text,) + (disable_btn,) * 4


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]
            + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return (
        states + [x.to_gradio_chatbot() for x in states] + [None] + [no_change_btn] * 6
    )


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_multimodal, invisible_text]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [enable_btn]
        + [""]
    )


def add_text(
    state0, state1, model_selector0, model_selector1, chat_input, request: gr.Request
):
    if isinstance(chat_input, dict):
        text, images = chat_input["text"], chat_input["files"]
    else:
        text = chat_input
        images = []

    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        if len(images) > 0:
            model_left, model_right = get_battle_pair(
                vl_models,
                VISION_BATTLE_TARGETS,
                VISION_OUTAGE_MODELS,
                VISION_SAMPLING_WEIGHTS,
                VISION_SAMPLING_BOOST_MODELS,
            )
            states = [
                State(model_left, is_vision=True),
                State(model_right, is_vision=True),
            ]
        else:
            model_left, model_right = get_battle_pair(
                text_models,
                BATTLE_TARGETS,
                OUTAGE_MODELS,
                SAMPLING_WEIGHTS,
                SAMPLING_BOOST_MODELS,
            )

            states = [
                State(model_left, is_vision=False),
                State(model_right, is_vision=False),
            ]

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None, ""]
            + [
                no_change_btn,
            ]
            * 7
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]

    images = convert_images_to_conversation_format(images)

    text, image_flagged, csam_flag = moderate_input(
        state0, text, text, model_list, images, ip
    )

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [{"text": CONVERSATION_LIMIT_MSG}, ""]
            + [
                no_change_btn,
            ]
            * 7
            + [""]
        )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [
                {
                    "text": IMAGE_MODERATION_MSG
                    + " PLEASE CLICK 🎲 NEW ROUND TO START A NEW CONVERSATION."
                },
                "",
            ]
            + [no_change_btn] * 7
            + [""]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(
            states[i], text, images, csam_flag=csam_flag
        )
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [disable_multimodal, visible_text]
        + [
            disable_btn,
        ]
        * 7
        + [hint_msg]
    )


def build_side_by_side_vision_ui_anony(text_models, vl_models, random_questions=None):
    notice_markdown = """
# ⚔️  LMSYS 聊天机器人竞技场(多模态): 现实环境下对LLM和VLM进行基准测试
[博客](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [论文](https://arxiv.org/abs/2403.04132) | [数据集](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) | [Kaggle 竞赛](https://www.kaggle.com/competitions/lmsys-chatbot-arena)


## 📜 规则
- 向两个匿名模型(如ChatGPT、Gemini、Claude、Llama)提出任何问题,并为更好的一个投票!
- 您可以继续聊天直到确定赢家。
- 如果在对话过程中透露了模型身份,投票将不被计入。
- **新功能** 图像支持: <span style='color: #DE3163; font-weight: bold'>在第一轮上传图像</span>即可解锁多模态竞技场! 图像应小于15MB。

## 🏆 聊天机器人竞技场 [排行榜](https://leaderboard.lmsys.org)
- 我们已收集**1,000,000+**人类投票,为100多个模型计算LLM Elo排行榜。在[这里](https://leaderboard.lmsys.org)查看谁是🥇LLM冠军!

## 👇 现在开始聊天!
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row():
        with gr.Column(scale=2, visible=False) as image_column:
            imagebox = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
            )

        with gr.Column(scale=5):
            with gr.Group(elem_id="share-region-anony"):
                with gr.Accordion(
                    f"🔍 Expand to see the descriptions of {len(text_models) + len(vl_models)} models",
                    open=False,
                ):
                    model_description_md = get_model_description_md(
                        text_models + vl_models
                    )
                    gr.Markdown(
                        model_description_md, elem_id="model_description_markdown"
                    )

                with gr.Row():
                    for i in range(num_sides):
                        label = "Model A" if i == 0 else "Model B"
                        with gr.Column():
                            chatbots[i] = gr.Chatbot(
                                label=label,
                                elem_id="chatbot",
                                height=650,
                                show_copy_button=True,
                            )

                with gr.Row():
                    for i in range(num_sides):
                        with gr.Column():
                            model_selectors[i] = gr.Markdown(
                                anony_names[i], elem_id="model_selector_md"
                            )
    with gr.Row():
        slow_warning = gr.Markdown("", elem_id="notice_markdown")

    with gr.Row():
        leftvote_btn = gr.Button(value="👈  A更好", visible=False, interactive=False)
        rightvote_btn = gr.Button(value="👉  B更好", visible=False, interactive=False)
        tie_btn = gr.Button(value="🤝  平局", visible=False, interactive=False)
        bothbad_btn = gr.Button(value="👎  都不好", visible=False, interactive=False)


    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 输入您的提示并按回车",
            elem_id="input_box",
            visible=False,
        )

        multimodal_textbox = gr.MultimodalTextbox(
            file_types=["image"],
            show_label=False,
            container=True,
            placeholder="在此输入您的提示或添加图片",
            elem_id="input_box",
        )
        # send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        if random_questions:
            global vqa_samples
            with open(random_questions, "r") as f:
                vqa_samples = json.load(f)

            random_btn = gr.Button(value="🔮 随机图片", interactive=True)
        clear_btn = gr.Button(value="🎲 新一轮", interactive=False)
        regenerate_btn = gr.Button(value="🔄  重新生成", interactive=False)
        share_btn = gr.Button(value="📷  分享")

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1800,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history,
        None,
        states
        + chatbots
        + model_selectors
        + [multimodal_textbox, textbox]
        + btn_list
        + [random_btn]
        + [slow_warning],
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    multimodal_textbox.input(add_image, [multimodal_textbox], [imagebox]).then(
        set_visible_image, [multimodal_textbox], [image_column]
    ).then(
        clear_history_example,
        None,
        states + chatbots + model_selectors + [multimodal_textbox, textbox] + btn_list,
    )

    multimodal_textbox.submit(
        add_text,
        states + model_selectors + [multimodal_textbox],
        states
        + chatbots
        + [multimodal_textbox, textbox]
        + btn_list
        + [random_btn]
        + [slow_warning],
    ).then(set_invisible_image, [], [image_column]).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states
        + chatbots
        + [multimodal_textbox, textbox]
        + btn_list
        + [random_btn]
        + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    if random_questions:
        random_btn.click(
            get_vqa_sample,  # First, get the VQA sample
            [],  # Pass the path to the VQA samples
            [multimodal_textbox, imagebox],  # Outputs are textbox and imagebox
        ).then(set_visible_image, [multimodal_textbox], [image_column]).then(
            clear_history_example,
            None,
            states
            + chatbots
            + model_selectors
            + [multimodal_textbox, textbox]
            + btn_list
            + [random_btn],
        )

    return states + model_selectors
