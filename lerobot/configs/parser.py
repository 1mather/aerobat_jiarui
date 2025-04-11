# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from functools import wraps
from pathlib import Path
from typing import Sequence

import draccus

from lerobot.common.utils.utils import has_method

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"
draccus.set_config_type("json")


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ['--env.discover_packages_path=my_package',
        ...         '--other_arg=value']
        >>> parse_plugin_args('discover_packages_path', args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")       # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg):
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def wrap(config_path: Path | None = None):
    """
    HACK: Similar to draccus.wrap but does three additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments. These plugins will typically register
            their own subclasses of config classes, so that draccus can find the right class to instantiate
            from the CLI '.type' arguments
    """

    def wrapper_outer(fn):
        """
        装饰器函数，包装原始函数以便自动处理配置解析和命令行参数。
        这个函数是一个闭包，返回内部的包装函数。
        
        参数:
            fn (callable): 要装饰的目标函数，通常是接受配置对象作为第一个参数的主函数。
        
        返回:
            callable: 包装后的函数，能够处理命令行参数并初始化配置对象。

        为什么要包装两层
        """
        @wraps(fn)  # 保留被装饰函数的元数据（如函数名、文档字符串等）
        def wrapper_inner(*args, **kwargs):
            """
            内部包装函数，实际处理参数解析和函数调用逻辑。
            
            参数:
                *args: 传递给原始函数的位置参数
                **kwargs: 传递给原始函数的关键字参数
                
            返回:
                任何类型: 原始函数的返回值
            """
            # 获取原始函数的参数规格
            argspec = inspect.getfullargspec(fn)
            # 获取第一个参数的类型注解，这应该是配置对象的类型
            argtype = argspec.annotations[argspec.args[0]]
            
            # 检查是否已经传入了配置对象
            if len(args) > 0 and type(args[0]) is argtype:
                # 如果第一个参数已经是正确类型的配置对象，直接使用它
                cfg = args[0]
                # 移除第一个参数，因为它将单独传递给原始函数
                args = args[1:]
            else:
                # 如果没有传入配置对象，需要从命令行参数创建一个
                
                # 获取命令行参数
                cli_args = sys.argv[1:]
                
                # 解析插件参数，这些是用于动态加载功能的特殊参数
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        # 尝试加载插件
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # 如果插件加载失败，添加相关命令行参数到错误信息中
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    # 从命令行参数中过滤掉已处理的插件参数
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                
                # 从命令行参数中解析配置文件路径
                config_path_cli = parse_arg("config_path", cli_args)
                
                # 检查配置类型是否有获取路径字段的方法
                if has_method(argtype, "__get_path_fields__"):
                    # 获取需要转换为Path对象的字段
                    path_fields = argtype.__get_path_fields__()
                    # 过滤掉路径相关的命令行参数
                    cli_args = filter_path_args(path_fields, cli_args)
                
                # 检查配置类型是否有从预训练加载的方法，且是否提供了配置路径
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    # 过滤掉配置路径参数，因为它将单独使用
                    cli_args = filter_arg("config_path", cli_args)
                    # 从预训练模型加载配置
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    # 使用draccus解析器创建配置对象
                    # config_path是一个未在代码片段中定义的变量，可能是全局变量或默认路径
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
                    """
                    
                    """
            
            # 使用解析好的配置对象调用原始函数
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
