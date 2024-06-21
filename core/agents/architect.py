from typing import Optional, List
from pydantic import BaseModel, Field
from core.agents.base import BaseAgent
from core.agents.convo import AgentConvo
from core.agents.response import AgentResponse
from core.db.models import Specification
from core.llm.parser import JSONParser
from core.log import get_logger
from core.telemetry import telemetry
from core.templates.example_project import EXAMPLE_PROJECTS
from core.templates.registry import PROJECT_TEMPLATES, ProjectTemplateEnum
from core.ui.base import ProjectStage

ARCHITECTURE_STEP_NAME = "Project architecture"
WARN_SYSTEM_DEPS = ["docker", "kubernetes", "microservices"]
WARN_FRAMEWORKS = ["next.js", "vue", "vue.js", "svelte", "angular"]
WARN_FRAMEWORKS_URL = "https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT-Pilot-with-frontend-frameworks"

log = get_logger(__name__)

class SystemDependency(BaseModel):
    name: str = Field(None, description="Name of the system dependency, e.g., Node.js or Python.")
    description: str = Field(None, description="One-line description of the dependency.")
    test: str = Field(None, description="Command line to test whether the dependency is available on the system.")
    required_locally: bool = Field(None, description="Whether this dependency must be installed locally.")

class PackageDependency(BaseModel):
    name: str = Field(None, description="Name of the package dependency, e.g., Express or React.")
    description: str = Field(None, description="One-line description of the dependency.")

class Architecture(BaseModel):
    architecture: str = Field(None, description="General description of the app architecture.")
    system_dependencies: List[SystemDependency] = Field(None, description="List of system dependencies required to build and run the app.")
    package_dependencies: List[PackageDependency] = Field(None, description="List of framework/language-specific packages used by the app.")
    template: Optional[ProjectTemplateEnum] = Field(None, description="Project template to use for the app, if any (optional).")

class Architect(BaseAgent):
    agent_type = "architect"
    display_name = "Architect"

    async def run(self) -> AgentResponse:
        await self.ui.send_project_stage(ProjectStage.ARCHITECTURE)
        spec = self.current_state.specification.clone()

        if spec.example_project:
            self.prepare_example_project(spec)
        else:
            await self.plan_architecture(spec)

        await self.check_system_dependencies(spec)

        self.next_state.specification = spec
        telemetry.set("template", spec.template)
        self.next_state.action = ARCHITECTURE_STEP_NAME
        return AgentResponse.done(self)

    async def plan_architecture(self, spec: Specification):
        await self.send_message("Planning project architecture ...")

        llm = self.get_llm()
        convo = AgentConvo(self).template("technologies", templates=PROJECT_TEMPLATES).require_schema(Architecture)
        arch: Architecture = await llm(convo, parser=JSONParser(Architecture))

        await self.check_compatibility(arch)

        spec.architecture = arch.architecture
        spec.system_dependencies = [d.model_dump() for d in arch.system_dependencies]
        spec.package_dependencies = [d.model_dump() for d in arch.package_dependencies]
        spec.template = arch.template.value if arch.template else None

    async def check_compatibility(self, arch: Architecture) -> bool:
        warn_system_deps = [dep.name for dep in arch.system_dependencies if dep.name.lower() in WARN_SYSTEM_DEPS]
        warn_package_deps = [dep.name for dep in arch.package_dependencies if dep.name.lower() in WARN_FRAMEWORKS]

        if warn_system_deps:
            await self.warn_about_dependencies(warn_system_deps, "system dependencies")

        if warn_package_deps:
            await self.warn_about_dependencies(warn_package_deps, "frameworks", WARN_FRAMEWORKS_URL)

        return True

    async def warn_about_dependencies(self, deps: List[str], dep_type: str, url: Optional[str] = None):
        message = f"Warning: GPT Pilot doesn't officially support {', '.join(deps)}. "
        message += f"You can try to use {'it' if len(deps) == 1 else 'them'}, but you may run into problems."
        if url:
            message += f" Visit {url} for more information."

        await self.ask_question(
            message,
            buttons={"continue": "Continue"},
            buttons_only=True,
            default="continue",
        )

    def prepare_example_project(self, spec: Specification):
        log.debug(f"Setting architecture for example project: {spec.example_project}")
        arch = EXAMPLE_PROJECTS[spec.example_project]["architecture"]

        spec.architecture = arch["architecture"]
        spec.system_dependencies = arch["system_dependencies"]
        spec.package_dependencies = arch["package_dependencies"]
        spec.template = arch["template"]
        telemetry.set("template", spec.template)

    async def check_system_dependencies(self, spec: Specification):
        deps = spec.system_dependencies

        for dep in deps:
            status_code, _, _ = await self.process_manager.run_command(dep["test"])
            dep["installed"] = status_code == 0
            if status_code != 0:
                remedy = "Please install it before proceeding." if dep["required_locally"] else "If you would like to use it locally, please install it before proceeding."
                await self.send_message(f"❌ {dep['name']} is not available. {remedy}")
                await self.ask_question(
                    f"Once you have installed {dep['name']}, please press Continue.",
                    buttons={"continue": "Continue"},
                    buttons_only=True,
                    default="continue",
                )
            else:
                await self.send_message(f"✅ {dep['name']} is available.")

        telemetry.set(
            "architecture",
            {
                "description": spec.architecture,
                "system_dependencies": deps,
                "package_dependencies": spec.package_dependencies,
            },
        )
