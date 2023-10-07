/*
 * File   : dsl_types.h
 * Author : Jiayuan Mao
 * Email  : maojiayuan@gmail.com
 * Date   : 09/09/2023
 *
 * This file is part of Project Concepts.
 * Distributed under terms of the MIT license.
 */

#pragma once

#include <string>

namespace concepts {

class TypeBase {
public:
  TypeBase(const std::string &name) : m_name(name) {}
  virtual ~TypeBase() = default;

  const std::string &name() const {
    return m_name;
  }

private:
  std::string m_name;
};

class ObjectType: public TypeBase {
};

class Variable {
public:
  Variable(const std::string &name, const std::string &type) : m_name(name), m_type(type) {}

  const std::string &name() const {
    return m_name;
  }

  const std::string &type() const {
    return m_type;
  }

private:
  std::string m_name;
  std::string m_type;
};

class FunctionType {
public:
  FunctionType(const std::string &return_type, const std::vector<std::string> &arg_types)
    : m_return_type(return_type), m_arg_types(arg_types) {}
  virtual ~FunctionType() = default;

  const std::string &return_type() const {
    return m_return_type;
  }
  const std::vector<std::string> &arg_types() const {
    return m_arg_types;
  }
  const std::string &arg_type(int i) const {
    return m_arg_types[i];
  }

private:
  std::string m_return_type;
  std::vector<std::string> m_arg_types;
};

class Function {
public:
  Function(const std::string &name, const FunctionType &type) : m_name(name), m_type(type) {}
  virtual ~Function() = default;

  const std::string &name() const {
    return m_name;
  }

  const FunctionType &type() const {
    return m_type;
  }

private:
  std::string m_name;
  FunctionType m_type;
};

} // namespace concepts
