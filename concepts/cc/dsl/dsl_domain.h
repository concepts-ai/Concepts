/*
 * File   : dsl_domain.h
 * Author : Jiayuan Mao
 * Email  : maojiayuan@gmail.com
 * Date   : 09/10/2023
 *
 * This file is part of Project Concepts.
 * Distributed under terms of the MIT license.
 */

#pragma once

#include <hashmap>
#include "dsl_types.h"

namespace concepts {

class Domain {
public:
  Domain() {}
  virtual ~Domain() = default;

  void add_type(const TypeBase &type) {
    m_types.insert({type.name(), type});
  }

  void add_function(const FunctionType &func) {
    m_functions.insert({func.name(), func});
  }

  const TypeBase &get_type(const std::string &name) const {
    return m_types.at(name);
  }

  const FunctionType &get_function(const std::string &name) const {
    return m_functions.at(name);
  }

private:
  std::hashmap<std::string, TypeBase> m_types;
  std::hashmap<std::string, FunctionType> m_functions;
};

class ExecutorContext {
public:
  ExecutorContext() {}
  virtual ~ExecutorContext() = default;

  void add_variable(const Variable &var) {
    m_variables.insert({var.name(), var});
  }

  const Variable &get_variable(const std::string &name) const {
    return m_variables.at(name);
  }

private:
  std::hashmap<std::string, Variable> m_variables;
};

} // namespace concepts
