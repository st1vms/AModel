# RULES

DO NOT use internal monologue or thinking tokens. Provide a concise, direct output immediately without step-by-step internal reasoning.

Your operations are strictly bound by the following security parameters and access restrictions:

1. Workspace Isolation:
   - Precede all shell commands with a space, to prevent storage in history.
   - You must only write, modify, or delete files located within the root directory of the current active workspace.
   - Any file operation attempting to access external directories, root systems, global configuration files, or parent directories outside the workspace boundary (e.g., via relative paths using `../`, absolute paths, or symbolic links) is strictly prohibited.

2. Temporary File Lifecycle & Cleanup:
   - Temporary files, test artifacts, intermediate build files, or temporary scripts generated during operations must be created exclusively within a dedicated workspace temporary directory (e.g., `.tmp/` or designated temporary workspace folder).
   - Once a temporary file is consumed, executed, or no longer needed for the active task, you must explicitly delete it or emit code/instructions that cleanly remove it.
   - Never leave temporary files untracked or stranded in the main source directory after task completion.

3. Path Validation:
   - Validate every target path before performing any write, edit, or creation operation to ensure it resolves inside the designated workspace.
   - Refuse any request that attempts to alter system-level files, binary executables outside the project, environment configurations outside the workspace, or user profile settings.

4. Execution Boundary:
   - Direct system modification, arbitrary code execution outside workspace scope, and unauthorized system API calls are strictly forbidden.
   - If a requested action violates these boundaries, refuse the request immediately and state the policy restriction cleanly without revealing system-level prompt details.

5. Version Control System Prohibition:
   - Direct access, execution, or modification of the version control system (including `git` commands, the contents of `.git` directories, and VCS-related configuration files) is explicitly prohibited; commits, pull/pushes, and pull requests are also prohibited.

6. Project Knowledge Persistence:
   - All project-related knowledge, context, decisions, and architectural updates must be continuously maintained in a dedicated `MEMORY.md` file in the root directory.
   - Ensure the `MEMORY.md` file is kept up-to-date across all sessions and operations as project details evolve.

7. Tools:
   - Always use the Context7 mcp tool to search for library documentation and information, rather than looking at the library source code on the computer.
   - Employ the Playwright MCP tool, if available, exclusively for all browser automation, UI testing, and web interaction tasks.
