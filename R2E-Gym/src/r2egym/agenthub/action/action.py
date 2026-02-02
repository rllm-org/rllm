import re
from typing import Dict
import shlex


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'file_editor')
      - parameters    (a dictionary of parameter_name -> value)

    Provides methods:
      - from_string(...) -> create Action from XML-like string
      - to_dict()        -> returns a JSON-like dict
      - to_bashcmd()     -> returns a string representing an equivalent bash command
    """

    def __init__(
        self, function_name: str, parameters: Dict[str, str], function_id: str = None
    ):
        self.function_name = function_name
        self.parameters = parameters
        # self.function_id = function_id

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """
        Parses a string of the form:

          <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
          </function>

        and returns an Action object.

        For example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>

        yields an Action with:
          function_name = "file_editor"
          parameters = {
            "command":  "view",
            "path":     "./sympy/tensor/array/dense_ndim_array.py",
            "concise":  "True"
          }
        """
        # Extract the function name: <function=...>
        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
        function_name = fn_match.group(1).strip() if fn_match else ""

        # Extract parameters of the form: <parameter=KEY>VALUE</parameter>
        # DOTALL allows the captured VALUE to span multiple lines
        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

        params = {}
        for param_key, param_value in param_matches:
            param_key = param_key.strip()
            param_value = param_value.strip()
            params[param_key] = param_value

        return cls(function_name, params)

    def __str__(self) -> str:
        return self.to_xml_string()

    def to_xml_string(self) -> str:
        """
        Returns an XML-like string representation of this action.

        Example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>
        """
        # Start with the function name
        xml_str = f"<function={self.function_name}>\n"

        # Add each parameter as <parameter=KEY>VALUE</parameter>
        for param_key, param_value in self.parameters.items():
            xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"

        xml_str += "</function>"
        return xml_str

    def to_dict(self) -> Dict[str, object]:
        """
        Returns a JSON-like dictionary representation of this action.

        Example:
          {
            "function": "file_editor",
            "parameters": {
              "command": "view",
              "path": "./sympy/tensor/array/dense_ndim_array.py",
              "concise": "True"
            }
          }
        """
        return {"function": self.function_name, "parameters": self.parameters}

    def to_bashcmd(self) -> str:
        """
        Converts this action into a Bash command string.

        Examples:
          If function_name == "execute_bash" and parameters = {
             "command": "search_dir",
             "search_term": "foo"
          }
          then this returns:
            execute_bash search_dir --search_term 'foo'

          If function_name == "file_editor" and parameters = {
             "command": "view",
             "path": "./some/path.py",
             "concise": "True"
          }
          then this returns:
            file_editor view --path './some/path.py' --concise 'True'
        """
        if not self.function_name:
            return ""
        elif self.function_name == "finish" or self.function_name == "submit":
            return "echo '<<<Finished>>>'"

        # Start building the command
        cmd_parts = [shlex.quote(self.function_name)]

        # If there's a 'command' parameter, put that next
        base_command = self.parameters.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        # Append all other parameters
        for param_key, param_value in self.parameters.items():
            if param_key == "command":
                continue

            # Safely quote the param_value
            param_value_quoted = shlex.quote(str(param_value))
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(param_value_quoted)

        return " ".join(cmd_parts)


if __name__ == "__main__":
    # Sample usage

    # Example 1
    xml_1 = """
    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
      <parameter=concise>True</parameter>
    </function>
    """
    action1 = Action.from_string(xml_1)
    print("[Example 1] Action as dict:", action1.to_dict())
    print("[Example 1] Action as bashcmd:", action1.to_bashcmd(), "\n")

    # Example 2
    xml_2 = """
    <function=execute_bash>
      <parameter=command>search_dir</parameter>
      <parameter=search_term>class ImmutableDenseNDimArray</parameter>
    </function>
    """
    action2 = Action.from_string(xml_2)
    print("[Example 2] Action as dict:", action2.to_dict())
    print("[Example 2] Action as bashcmd:", action2.to_bashcmd())
