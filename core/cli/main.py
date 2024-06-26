import sys
from argparse import Namespace
from asyncio import run
from typing import Tuple, Optional

from core.agents.orchestrator import Orchestrator
from core.cli.helpers import delete_project, init, list_projects, list_projects_json, load_project, show_config
from core.config import LLMProvider, get_config
from core.db.session import SessionManager
from core.db.v0importer import LegacyDatabaseImporter
from core.llm.base import APIError, BaseLLMClient
from core.log import get_logger
from core.state.state_manager import StateManager
from core.telemetry import telemetry
from core.ui.base import UIBase, UIClosedError, UserInput, pythagora_source

log = get_logger(__name__)

async def run_project(sm: StateManager, ui: UIBase) -> bool:
    telemetry.set("app_id", str(sm.project.id))
    telemetry.set("initial_prompt", sm.current_state.specification.description)

    orca = Orchestrator(sm, ui)
    try:
        success = await orca.run()
        telemetry.set("end_result", "success:exit" if success else "failure:api-error")
        return success
    except (KeyboardInterrupt, UIClosedError):
        log.info("Interrupted by user")
        telemetry.set("end_result", "interrupt")
        await sm.rollback()
    except APIError as err:
        log.warning(f"LLM API error occurred: {err.message}")
        await ui.send_message(
            f"Stopping Pythagora due to an error while calling the LLM API: {err.message}",
            source=pythagora_source,
        )
        telemetry.set("end_result", "failure:api-error")
        await sm.rollback()
    except Exception as err:
        log.error(f"Uncaught exception: {err}", exc_info=True)
        stack_trace = telemetry.record_crash(err)
        await sm.rollback()
        await ui.send_message(
            f"Stopping Pythagora due to error:\n\n{stack_trace}",
            source=pythagora_source,
        )
    return False

async def llm_api_check(ui: UIBase) -> bool:
    config = get_config()
    async def handler(*args, **kwargs):
        pass

    success = True
    checked_llms: set[LLMProvider] = set()
    for llm_config in config.all_llms():
        if llm_config.provider in checked_llms:
            continue

        client_class = BaseLLMClient.for_provider(llm_config.provider)
        llm_client = client_class(llm_config, stream_handler=handler, error_handler=handler)
        try:
            resp = await llm_client.api_check()
            if not resp:
                success = False
                log.warning(f"API check for {llm_config.provider.value} failed.")
            else:
                log.info(f"API check for {llm_config.provider.value} succeeded.")
        except APIError as err:
            await ui.send_message(
                f"API check for {llm_config.provider.value} failed with: {err}",
                source=pythagora_source,
            )
            log.warning(f"API check for {llm_config.provider.value} failed with: {err}")
            success = False

    if not success:
        telemetry.set("end_result", "failure:api-error")

    return success

async def start_new_project(sm: StateManager, ui: UIBase) -> bool:
    try:
        user_input = await ui.ask_question(
            "What is the project name?",
            allow_empty=False,
            source=pythagora_source,
        )
    except (KeyboardInterrupt, UIClosedError):
        return False

    if user_input.cancelled:
        return False

    project_state = await sm.create_project(user_input.text)
    return project_state is not None

async def run_pythagora_session(sm: StateManager, ui: UIBase, args: Namespace) -> bool:
    if not await llm_api_check(ui):
        await ui.send_message(
            "Pythagora cannot start because the LLM API is not reachable.",
            source=pythagora_source,
        )
        return False

    if args.project or args.branch or args.step:
        telemetry.set("is_continuation", True)
        success = await load_project(sm, args.project, args.branch, args.step)
        if not success:
            return False
    else:
        success = await start_new_project(sm, ui)
        if not success:
            return False

    return await run_project(sm, ui)

async def async_main(ui: UIBase, db: SessionManager, args: Namespace) -> bool:
    if args.list:
        await list_projects(db)
        return True
    elif args.list_json:
        await list_projects_json(db)
        return True
    if args.show_config:
        show_config()
        return True
    elif args.import_v0:
        importer = LegacyDatabaseImporter(db, args.import_v0)
        await importer.import_database()
        return True
    elif args.delete:
        success = await delete_project(db, args.delete)
        return success

    telemetry.set("user_contact", args.email)
    if args.extension_version:
        telemetry.set("is_extension", True)
        telemetry.set("extension_version", args.extension_version)

    sm = StateManager(db, ui)
    ui_started = await ui.start()
    if not ui_started:
        return False

    telemetry.start()
    success = await run_pythagora_session(sm, ui, args)
    await telemetry.send()
    await ui.stop()

    return success

def run_pythagora() -> int:
    ui, db, args = init()
    if not ui or not db:
        return -1
    success = run(async_main(ui, db, args))
    return 0 if success else -1

if __name__ == "__main__":
    sys.exit(run_pythagora())
