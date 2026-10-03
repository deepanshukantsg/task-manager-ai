from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, text
import streamlit as st
import os

# load the env
load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            google_sub TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tasks (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT
                CHECK (status IN ('pending', 'in_progress', 'completed'))
                DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """))

def get_current_user_id():
    with engine.begin() as conn:
        user_id = conn.execute(
            text("""
                INSERT INTO users (
                    google_sub,
                    email,
                    name
                )
                VALUES (
                    :google_sub,
                    :email,
                    :name
                )
                ON CONFLICT (google_sub)
                DO UPDATE SET
                    email = EXCLUDED.email,
                    name = EXCLUDED.name
                RETURNING id
            """),
            {
                "google_sub": st.user["sub"],
                "email": st.user["email"],
                "name": st.user.get("name", "")
            }
        ).scalar_one()

    return user_id



if not st.user.is_logged_in:
    st.title("AI Task Manager")
    st.write("Login to manage your tasks.")

    st.button(
        "Continue with Google",
        on_click=st.login
    )

    st.stop()


current_user_id = get_current_user_id()

st.title("Welcome to my task manager!")
st.write(f"Welcome, {st.user.name}")

st.sidebar.write(st.user.email)

st.sidebar.button(
    "Logout",
    on_click=st.logout
)


# creating llm
llm = ChatOpenAI(model='gpt-4', streaming=True)

# custom tools creation
def get_task_tools(user_id):

    @tool
    def create_task(title: str, description: str = "") -> str:
        """Create a task for the logged-in user."""

        with engine.begin() as conn:
            task = conn.execute(
                text("""
                    INSERT INTO tasks (
                        user_id,
                        title,
                        description,
                        status
                    )
                    VALUES (
                        :user_id,
                        :title,
                        :description,
                        'pending'
                    )
                    RETURNING id, title
                """),
                {
                    "user_id": user_id,
                    "title": title,
                    "description": description
                }
            ).mappings().one()

        return f"Task created: {task['title']}."


    @tool
    def list_tasks() -> str:
        """List all tasks belonging to the logged-in user."""

        with engine.begin() as conn:
            tasks = conn.execute(
                text("""
                    SELECT id, title, description, status
                    FROM tasks
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                """),
                {"user_id": user_id}
            ).mappings().all()

        if not tasks:
            return "You have no tasks."

        return "\n".join(
            f"{task['id']}. {task['title']} [{task['status']}]"
            for task in tasks
        )


    @tool
    def update_task_status(task_id: int, status: str) -> str:
        """Change the status of one of the logged-in user's tasks."""

        allowed = {"pending", "in_progress", "completed"}

        if status not in allowed:
            return "Invalid task status."

        with engine.begin() as conn:
            task = conn.execute(
                text("""
                    UPDATE tasks
                    SET status = :status
                    WHERE id = :task_id
                      AND user_id = :user_id
                    RETURNING title
                """),
                {
                    "status": status,
                    "task_id": task_id,
                    "user_id": user_id
                }
            ).scalar_one_or_none()

        if task is None:
            return "Task not found."

        return f"Updated task: {task} → {status}"


    @tool
    def delete_task(task_id: int) -> str:
        """Delete one of the logged-in user's tasks."""

        with engine.begin() as conn:
            task = conn.execute(
                text("""
                    DELETE FROM tasks
                    WHERE id = :task_id
                      AND user_id = :user_id
                    RETURNING title
                """),
                {
                    "task_id": task_id,
                    "user_id": user_id
                }
            ).scalar_one_or_none()

        if task is None:
            return "Task not found."

        return f"Deleted task: {task}"


    return [
        create_task,
        list_tasks,
        update_task_status,
        delete_task
    ]


if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history=[]

system_prompt = """
You are a dedicated AI Task Manager.

Your ONLY purpose is to help the currently logged-in user manage their own tasks.

You may only perform task-management actions through the tools provided to you.

You may:
- Create tasks
- List the user's tasks
- Change task status
- Delete tasks
- Answer questions about the user's own tasks

You must NOT answer unrelated questions.

If the user asks about anything unrelated to task management, respond briefly:

"I can only help you manage your tasks."


TASK DATA

Each task contains:

- id: unique task ID
- user_id: owner of the task
- title: task name/title
- description: optional task description
- status: one of:
    - pending
    - in_progress
    - completed
- created_at: time the task was created


USER SECURITY RULES

1. The currently logged-in user's user_id is handled internally by the tools.
2. Never ask the user for their user_id.
3. Never invent or modify a user_id.
4. Never attempt to access another user's tasks.
5. Only use the task tools provided to you.
6. Do not generate or execute your own SQL.
7. Do not claim that an operation succeeded unless the tool confirms it succeeded.


TASK CREATION

When the user asks to create a task, use the create_task tool.

Examples:

"Create a task to fix the careers page"

Create:

title = "Fix the careers page"
description = ""

"Add a task called complete homepage animations"

Create:

title = "Complete homepage animations"

New tasks must be created with status:

pending

Do not ask for confirmation before creating a normal task.

If the user provides a description, pass it to the tool.

If the user does not provide a description, use an empty description.


TASK LISTING

When the user asks things like:

- "Show my tasks"
- "What tasks do I have?"
- "List my todos"
- "What is pending?"
- "What am I working on?"

Use the list_tasks tool.

Only discuss tasks returned by the tool.


TASK STATUS

Use the update_task_status tool when the user wants to change a task's status.

Interpret:

"complete task 3"
"finish task 3"
"mark task 3 done"

as:

status = "completed"

Interpret:

"start task 3"
"start working on task 3"
"mark task 3 in progress"

as:

status = "in_progress"

Interpret:

"move task 3 back to pending"
"mark task 3 pending"

as:

status = "pending"

Valid statuses are only:

- pending
- in_progress
- completed

Never invent another status.


TASK DELETION

Use the delete_task tool when the user asks to delete a specific task.

Examples:

"Delete task 4"
"Remove task 2"

Do not ask for confirmation when a single task is clearly identified.

If the requested task does not exist, report that briefly.


AMBIGUOUS REQUESTS

If the user asks to update, complete, or delete a task but does not provide enough information to identify the task, ask only for the missing task information.

Do not guess task IDs.


RESPONSES

Keep responses short and task-focused.

Examples:

After creation:

"Task created: Fix the responsive design of the Careers page of Daon."

After completion:

"Task marked as completed: Fix the responsive design of the Careers page of Daon."

After moving to in progress:

"Task marked as in progress: Fix the responsive design of the Careers page of Daon."

After deletion:

"Task deleted: Fix the responsive design of the Careers page of Daon."

Do not explain internal tools, database logic, SQL, user IDs, authentication, or implementation details unless the user explicitly asks about them.

Do not pretend that a task exists if the tools do not return it.

Do not perform any action outside task management.
"""

@st.cache_resource
def getAgent(user_id):
    agent = create_agent(
            model=llm,  
            tools=get_task_tools(user_id),
            system_prompt=system_prompt,
            checkpointer=MemorySaver(),
    )
    return agent

agent = getAgent(current_user_id)

if 'messages' not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

prompt = st.chat_input("What's your task query? What do you want to manage?")

if(prompt):

    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})

    with st.chat_message('ai'):
        with st.spinner('Spinner...'):
            response = agent.invoke(
                    {"messages":[{"role":"user", "content": prompt}]},
                    {"configurable": {"thread_id": f"user:{st.user['sub']}:main"}}
                )
            result = response['messages'][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai", "content":result})


